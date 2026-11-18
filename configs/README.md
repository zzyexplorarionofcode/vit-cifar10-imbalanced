# 配置说明

本实验不依赖 YAML 配置文件；每个实验的完整命令行参数、随机种子、数据划分和超参数都会随结果写入
`results/<tag>.json` 的 `config` 字段，checkpoint 中也保存了 `model_kwargs`，可直接用于复现和
复杂度测量。常用命令见项目根目录的 `README.md` 和 `scripts/run_all.ps1`。
