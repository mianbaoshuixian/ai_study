# Memory Index

- [Jupyter notebook JSON 中文引号问题](.comate/memory/feedback_jupyter_json_quotes.md) — 生成 .ipynb 时禁用中文弯引号，改用「」，否则 JSON 解析报错
- [课件图片规范 - 尚硅谷图示](.comate/memory/project_lesson_images.md) — 生成 lesson.ipynb 后提醒用户在核心公式/架构章节补充尚硅谷截图，图放公式正上方作为视觉锚点

<svg viewBox="0 0 480 400" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Microsoft YaHei, sans-serif">
  <rect x="0" y="0" width="480" height="400" fill="#fbfbfd"/>
  <!-- grid -->
  <g stroke="#e8e8ee" stroke-width="1">
    <line x1="40" y1="360" x2="440" y2="360"/>
    <line x1="80" y1="40" x2="80" y2="360"/>
  </g>
  <!-- parallelogram fill -->
  <polygon points="80,320 280,320 360,160 160,160" fill="#4c8bf5" fill-opacity="0.10" stroke="#c5d6fb" stroke-width="1"/>
  <!-- opposite edges (dashed, = same vectors translated) -->
  <line x1="280" y1="320" x2="360" y2="160" stroke="#4c8bf5" stroke-width="2" stroke-dasharray="5 4" opacity="0.6"/>
  <line x1="160" y1="160" x2="360" y2="160" stroke="#e5484d" stroke-width="2" stroke-dasharray="5 4" opacity="0.6"/>
  <!-- long diagonal a+b -->
  <line x1="80" y1="320" x2="360" y2="160" stroke="#2fa84f" stroke-width="3"/>
  <!-- short diagonal -->
  <line x1="280" y1="320" x2="160" y2="160" stroke="#b48a00" stroke-width="2" stroke-dasharray="2 4"/>
  <!-- vector a -->
  <defs>
    <marker id="ar" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#333"/>
    </marker>
    <marker id="arR" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#e5484d"/></marker>
    <marker id="arB" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#4c8bf5"/></marker>
    <marker id="arG" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#2fa84f"/></marker>
  </defs>
  <line x1="80" y1="320" x2="280" y2="320" stroke="#e5484d" stroke-width="3" marker-end="url(#arR)"/>
  <line x1="80" y1="320" x2="160" y2="160" stroke="#4c8bf5" stroke-width="3" marker-end="url(#arB)"/>
  <line x1="80" y1="320" x2="360" y2="160" stroke="#2fa84f" stroke-width="3" marker-end="url(#arG)"/>
  <!-- points -->
  <circle cx="80" cy="320" r="3.5" fill="#333"/>
  <!-- labels -->
  <text x="180" y="342" fill="#e5484d" font-size="16" font-style="italic">a</text>
  <text x="108" y="240" fill="#4c8bf5" font-size="16" font-style="italic">b</text>
  <text x="235" y="228" fill="#2fa84f" font-size="16" font-weight="bold">a + b（长对角线）</text>
  <text x="205" y="255" fill="#b48a00" font-size="13">a − b（短对角线）</text>
  <text x="60" y="338" fill="#333" font-size="13">O</text>
</svg>

<svg viewBox="0 0 480 320" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Microsoft YaHei, sans-serif">
  <rect x="0" y="0" width="480" height="320" fill="#fbfbfd"/>
  <defs>
    <marker id="a2R" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#e5484d"/></marker>
    <marker id="a2B" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#4c8bf5"/></marker>
  </defs>
  <!-- sun rays (perpendicular projection direction) -->
  <g stroke="#f5b301" stroke-width="1.5" stroke-dasharray="3 4" opacity="0.8">
    <line x1="160" y1="120" x2="160" y2="250"/>
  </g>
  <!-- vector a (base) -->
  <line x1="80" y1="250" x2="360" y2="250" stroke="#e5484d" stroke-width="3" marker-end="url(#a2R)"/>
  <!-- vector b -->
  <line x1="80" y1="250" x2="160" y2="120" stroke="#4c8bf5" stroke-width="3" marker-end="url(#a2B)"/>
  <!-- projection segment (the shadow) -->
  <line x1="80" y1="250" x2="160" y2="250" stroke="#2fa84f" stroke-width="6" opacity="0.85"/>
  <!-- right angle mark -->
  <path d="M160,238 L172,238 L172,250" fill="none" stroke="#f5b301" stroke-width="1.5"/>
  <!-- angle theta -->
  <path d="M120,250 A40,40 0 0 0 108,224" fill="none" stroke="#888" stroke-width="1.2"/>
  <text x="122" y="238" fill="#888" font-size="13">θ</text>
  <!-- points & labels -->
  <circle cx="80" cy="250" r="3.5" fill="#333"/>
  <text x="300" y="242" fill="#e5484d" font-size="16" font-style="italic">a</text>
  <text x="120" y="180" fill="#4c8bf5" font-size="16" font-style="italic">b</text>
  <text x="95" y="272" fill="#2fa84f" font-size="13" font-weight="bold">b 在 a 上的投影 = |b|cosθ</text>
  <text x="60" y="268" fill="#333" font-size="13">O</text>
  <text x="120" y="300" fill="#333" font-size="14">点积 a·b = |a| ×（绿色影子长度）</text>
</svg>