# 一键复现脚本：按顺序完成数据预处理、四个模型训练、复杂度测量、图表和报告。
# 运行方式（在项目根目录）：powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
# 注意：完整训练在 CPU 上约需 1 小时；只想冒烟测试可先手动运行带 --limit-train 的小命令。

# 任何一步失败就立刻停止，避免后面的步骤读到不完整的结果。
$ErrorActionPreference = "Stop"
$python = "python"
$threads = 8

function Invoke-Step {
    # 统一封装“打印标题 -> 执行 Python -> 检查退出码”。
    param([string]$Label, [string[]]$ArgList)
    Write-Host "`n===== $Label =====" -ForegroundColor Cyan
    # @ArgList 是 PowerShell 的 splatting 语法：把数组展开成多个命令行参数。
    & $python @ArgList
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}

# 数据只需转换一次；如果已经存在 npy 就跳过。
if (-not (Test-Path "data\train_images.npy")) {
    Invoke-Step "Prepare data" @("scripts\prepare_data.py")
}

# 基线：普通交叉熵与类别平衡损失，二者只差损失函数。
Invoke-Step "ViT + CE" @(
    "scripts\train.py", "--model", "vit", "--loss", "ce", "--tag", "baseline_ce",
    "--epochs", "20", "--num-threads", "$threads", "--save-predictions"
)
Invoke-Step "ViT + CB" @(
    "scripts\train.py", "--model", "vit", "--loss", "cb", "--tag", "baseline_cb",
    "--epochs", "20", "--num-threads", "$threads", "--export-logits", "--save-predictions"
)

# 轻量模型：用 CB 基线的缓存 logits 做蒸馏，因此必须在 baseline_cb 之后运行。
Invoke-Step "DynamicViT + CB" @(
    "scripts\train.py", "--model", "dynamic", "--loss", "cb", "--tag", "dynamic_cb",
    "--epochs", "20", "--num-threads", "$threads",
    "--prune-after", "0,1,2", "--keep-ratios", "0.75,0.5,0.25",
    "--distill-logits", "checkpoints\baseline_cb_train_logits.pt", "--save-predictions"
)
Invoke-Step "MobileViT-Lite + CB" @(
    "scripts\train.py", "--model", "mobile", "--loss", "cb", "--tag", "mobile_cb",
    "--epochs", "20", "--num-threads", "$threads", "--dim", "72",
    "--mobile-depth", "2", "--global-depth", "1", "--mobile-patch", "4",
    "--distill-logits", "checkpoints\baseline_cb_train_logits.pt", "--save-predictions"
)

# 结果侧：FLOPs/延迟、逐阶段 token 数、图表与 Markdown 汇总、DOCX 报告。
Invoke-Step "Complexity" @("scripts\measure_complexity.py", "--threads", "$threads")
Invoke-Step "Token statistics" @("scripts\dynamic_tokens.py")
Invoke-Step "Code roadmap" @("scripts\make_code_roadmap.py")
Invoke-Step "Figures and tables" @("scripts\analyze.py")
Invoke-Step "Report" @("scripts\build_report.py")
Write-Host "`nAll done. See report\CIFAR10_ViT实验报告.docx" -ForegroundColor Green
