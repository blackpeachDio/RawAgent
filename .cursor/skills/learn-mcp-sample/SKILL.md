---
name: learn-mcp-sample
description: >-
  Demonstrates how to call Cursor MCP tools for simple utility tasks. Use when the user asks for
  current time, asks to echo text, or mentions MCP tool calling.
---

# Learn MCP Tool Calling (Sample)

## What this skill does
- Helps the agent decide to call an MCP tool when the user asks for simple utility results.

## When to use
Use this skill when the user requests one of the following:
- “现在几点/当前时间/获取当前时间”
- “回显/echo/把我说的话原样返回”
- The user explicitly mentions MCP tool calling.

## Step-by-step workflow (what to do and why)
1. Parse the user intent into one of the tool categories (`get_current_time`, `echo`).
   - Why: MCP tools require an explicit tool name and a structured argument object.

2. Call the matching MCP tool with the correct input fields.
   - Why: Tool calling is the reliable way to produce deterministic utility outputs.

3. Return only the tool result to the user, without inventing extra data.
   - Why: For utility tasks, users expect the raw tool output rather than a rewritten answer.

## Tool mapping (for the demo MCP server)
- `get_current_time`:
  - Input: `{ "timezone": "Asia/Shanghai" }` (timezone optional)
- `echo`:
  - Input: `{ "text": "..." }`

