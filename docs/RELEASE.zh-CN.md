# 新项目开源发布检查单

此包包含可导入的 Python package、CLI、单元测试、CI、LICENSE 和双语 README。当前是一个**本地可运行的开源原型**，尚未创建你的 GitHub 远程仓库，也没有发布 GitHub Release。

## 发布前

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[hf,dev]'
pytest -q
python -m compileall -q src examples
wspace demo --out runs/demo
wspace scan --traces runs/demo/traces.jsonl --out runs/demo/diagnosis
```

确认 `README` 中所有“真实模型实验”边界描述准确。不要提交 `runs/` 目录中的真实客户提示、API 返回、实验原始权重和未经许可的数据集。检查 `LICENSE`、原作者引用及 HF 模型许可证。

## 选择新的 GitHub 仓库名称

建议名称：`weightspace-doctor`。若本地已安装 GitHub CLI 且有对应 GitHub 权限，可以在解压目录手动执行：

```bash
git init
git add .
git commit -m "feat: initial weight-space diagnostics and reversible repair prototype"
git branch -M main
gh repo create weightspace-doctor --public --source=. --remote=origin --push
```

也可以先在 GitHub 网站创建空仓库，再配置 `git remote add origin` 和 `git push -u origin main`。

**注意**：上述是给维护者自己执行的发布指令；本项目并未自动执行、未擅自访问任何已有个人仓库，也不声称已打 tag 或上传 PyPI。

## 第一个开源版本

建议第一次公开标签为 `v0.1.0`，发布说明强调：

- 可用的 CLI 命令与 JSON/Markdown 报告。
- 仅支持未量化 HF 浮点权重编辑；支持对 MLX 4-bit 的 API 侧诊断，但不允许直接编辑量化权重。
- 实际测试基于小型 CPU 合成模型；未复现 Qwen3.5-0.8B 论文的测试成绩。
- 所有编辑候选必须经独立验证和人工审批才能用于生产环境。

## 本地准备记录（2026-11-03）

已进行 CPU 单元/集成测试（38 passed），并在本地准备 Git 仓库、初始 commit 和 `v0.1.0` 标签。**这不代表已创建或推送到远程 GitHub。** 需要先在 `h-yu-u` 账户创建空的 `weightspace-doctor` 仓库，再推送或用已连接的 GitHub 工具提交源码。GitHub Release 仍需在远程创建。

## 一键发布脚本（在具备 GitHub 身份验证的本机执行）

本项目还提供 `scripts/publish_github.sh`：检测 GitHub CLI 登录态；若 `h-yu-u/weightspace-doctor` 不存在，就创建**公开**仓库；安全校验远程地址后推送 main 与 v0.1.0，并创建 Release。它不会打印访问令牌；不会强制推送，也不会重写已有 origin。

```bash
brew install gh
gh auth login
bash scripts/publish_github.sh
```

仓库名可以通过 `GITHUB_REPO=xxx` 覆盖，前提是 GitHub 上确实由维护者控制。发布动作只会在你显式运行该脚本后发生。
