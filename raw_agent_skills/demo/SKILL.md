---
name: demo
description: 示例技能，演示 raw_agent_skills 目录结构与 frontmatter。
---

# Demo 技能

## 何时使用

当用户询问本仓库「技能机制」或需要遵循本示例中的简短约定时使用。

## 约定

- 每个技能一个子目录，根文件名为 `SKILL.md`。
- 通过工具 `get_agent_skill` 按目录名拉取正文，勿把大段说明写进工具描述里。
