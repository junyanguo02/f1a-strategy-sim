# PITWALL

从 web/ 运行。使用项目既有 Node/npm 环境与锁定的依赖。

```sh
npm run dev -- --host 127.0.0.1
npm test
npm run build
npm run preview -- --host 127.0.0.1
```

静态构建位于 dist/。默认相对资源路径，可通过 `PITWALL_BASE=/project/ npm run build` 指定站点前缀。三个页面使用哈希路由（#/、#/story、#/lab）。首次访问为英文，右上角切换语言后保存在本机。

- 首页与叙事比赛回放按需读取真实逐圈时间线，参照默认是冠军全程平均圈速。
- 第一章图表读取 findings/stop-counts.json 中的多车赛前回测车次计数。第二章从西班牙赛后导出的联合后验计算三种配方的用时项及逐胎龄中央 90% 区间。
- 实验室使用赛前 Python 热图与预设策略。切换赛前/赛后输入、情景、进站圈时，浏览器 Worker 在所选配置下比较策略。首选方案与其他候选统计上并列时展示导出的候选标签。甘特图把手支持拖动、方向键、Home、End。
- public/data/mock_* 仅供历史合约测试。生产包只复制真实导出与研究统计，模拟器逐抽样对照 fixture 仅供 Vitest。
- 从 web/ 运行 `PYTHONDONTWRITEBYTECODE=1 ../.venv/bin/python verification/audit.py` 可重新生成 TOST、逐圈开始频率及进站损失分解。verification/random-audit.json 保存本轮输入摘要和结果。
