# blog

技术博客仓库：文章 + 写作 skill + 自动部署。仓库地址 `pex_org/blog`（私有）。

## 这是什么

| 内容 | 说明 |
|---|---|
| 文章 | 根目录的 `.md`（草稿/源码）+ `.html`（渲染版，蓝白简约主题） |
| `index.html` | 目录页，每发一篇 HTML 文章在此加入口 |
| `SKILL.md` | 写作方法论（blog-skill），完整写作流程与配色规范 |
| 本文件 | 仓库首页说明 |

## 🔄 自动部署（push 后网页自动更新）

这个仓库接入了自动部署：**main 分支更新后，网页博客会自动更新，无需任何手动操作。**

### 工作流

```
改文章 → 建分支 → push → 提 PR → san-tian approve + merge
                                          │
                     main 更新 ──► Gitea webhook（push 事件）
                                          │
                     NAS blog-serve 容器收到 webhook → git pull --ff-only
                                          │
                              网页静态文件更新，立即生效
```

1. **改动必须走 PR**：main 有分支保护（禁止直接 push），流程是建分支 → 提 PR → `san-tian` review 后 merge。
2. **merge 即触发部署**：PR merge 到 main 后，Gitea 向 `blog-serve` 容器的 webhook 端点发 push 通知。
3. **自动 pull**：容器收到 webhook 后执行 `git pull --ff-only`，拉取最新 main。
4. **立即生效**：容器直接托管 `git pull` 后的目录，无需重启、无需拷贝。

### 技术栈

- **静态托管**：NAS 上的 Docker 容器 `blog-serve`（Python 静态服务器 + webhook 一体，监听 8081）
- **自动更新**：Gitea webhook（`push` 事件）→ 容器内 `git pull --ff-only`
- **源目录**：NAS `/vol1/1000/blog`（本仓库的 clone）
- **容器维护**：`docker restart blog-serve`；已设 `--restart unless-stopped`（NAS 重启自动恢复）

## 访问方式

| 网络 | 地址 |
|---|---|
| tailscale 内网 | http://100.78.161.108:8081/ |
| 公网 | 见 PassNAT 穿透（8081 端口映射） |

## 写作流程（摘要）

详见 `SKILL.md`。核心几条：

1. 草稿用 Markdown（`.md`），需展示时生成 HTML（蓝白主题，inline CSS 自包含）
2. 文件名 kebab-case 英文：`pd-decode-kvcache-offload.md`
3. 只写最终正确结论，不写过程叙事；数据来源标注在对应表格底下
4. 每发一篇 HTML，在 `index.html` 加目录入口
5. 诚实标注不确定处：`✓ 源码确认` / `✗ 纠正` / `⚠ 诚实标注`

## 目录结构

```
blog/
├── README.md          # 本文件（仓库首页）
├── SKILL.md           # 写作 skill（完整方法论）
├── index.html         # 目录页
├── *.md / *.html      # 文章
└── ...
```
