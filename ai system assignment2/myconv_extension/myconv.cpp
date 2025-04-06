#include <torch/extension.h>
#include <tuple>

// 前向传播（包含卷积+池化）
std::vector<torch::Tensor> myconv_forward(
    torch::Tensor input,         // [N, C_in, H, W]
    torch::Tensor weight,        // [C_out, C_in, K, K]
    int64_t stride,
    int64_t padding)
{
    // 参数检查
    TORCH_CHECK(input.dim() == 4, "输入必须是4维张量 (N, C_in, H, W)");
    TORCH_CHECK(weight.dim() == 4, "权重必须是4维张量 (C_out, C_in, K, K)");

    // 内存连续性保证
    input = input.contiguous();
    weight = weight.contiguous();

    // 1. 执行卷积操作
    auto conv_out = torch::conv2d(
        input,
        weight,
        /*bias=*/torch::Tensor(),
        stride,
        padding
    );

    // 2. 执行2x2 MaxPooling（步幅=2）
    auto pool_result = torch::max_pool2d_with_indices(
        conv_out,
        /*kernel_size=*/{2, 2},
        /*stride=*/{2, 2}
    );

    auto pool_out = std::get<0>(pool_result); // 池化输出
    auto indices = std::get<1>(pool_result);  // 池化索引

return {input,pool_out, conv_out, indices};
}

// 反向传播
std::vector<torch::Tensor> myconv_backward(
    torch::Tensor grad_pool,    // 池化层梯度 [N, C_out, H_pool, W_pool]
    torch::Tensor conv_out,     // 前向传播的卷积输出
    torch::Tensor pool_indices, // 池化索引
    torch::Tensor input,        // 原始输入
    torch::Tensor weight,       // 卷积核
    int64_t stride,
    int64_t padding)
{
    // 1. 计算池化层梯度
    auto grad_conv = torch::max_pool2d_with_indices_backward(
        grad_pool,
        conv_out,
        {2, 2},                // kernel_size
        {2, 2},                // stride
        {0, 0},                // padding
        {1,1},
        false,                 // ceil_mode
        pool_indices
    );

    // 2. 计算卷积层梯度
    // 输入梯度
    auto grad_input = torch::conv_transpose2d(
        grad_conv,
        weight,
        /*bias=*/torch::Tensor(),
        stride,
        padding
    );

    // 权重梯度
    auto grad_weight = torch::conv2d(
        input.transpose(0, 1),  // [C_in, N, H, W]
        grad_conv.transpose(0, 1), // [C_out, N, H_conv, W_conv]
        /*bias=*/torch::Tensor(),
        /*stride=*/1,
        padding,
        1           // groups=输入通道数
    ).transpose(0, 1);          // 恢复为[C_out, C_in, K, K]

    return {grad_input, grad_weight};
}

// 绑定为PyTorch模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &myconv_forward, "Conv2D+MaxPool forward");
    m.def("backward", &myconv_backward, "Conv2D+MaxPool backward");
}