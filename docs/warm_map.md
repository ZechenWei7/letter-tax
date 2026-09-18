# B_warm 控制词 → 符号映射（v7 §4，冻结）

所有符号在 Qwen3 词表里"带前导空格"与"不带"都是**单个**白名单 token（无字母、无 U+FFFD 字节碎片），2026-09-18 用 unsloth/Qwen3-1.7B 分词器核对（与 Qwen3-4B 同词表）。
被排除的候选：∴ ⊢ ⊥ ⊤ ✗ ‖ ∨ ≡ ⟹ ∅ † ‡ ◇ □ ∵ ⊗ ⊕ ÷ ∞ √（带前导空格时切成字节碎片 → 禁集）。

| 控制词 | 符号 | 角色（strategy.py） |
|---|---|---|
| if | ¿ | case_splits |
| assume, suppose | » | case_splits |
| case | § | case_splits |
| then | → | propagation |
| so, therefore, thus, hence | ⇒ | propagation |
| contradiction, contradicts, impossible | × | contradiction |
| consistent | ✓ | consistent |
| else, otherwise | « | — |
| but | ~ | — |
| and / or / not | & / \| / ! | — |
| knight / knave | 1 / 0 | — |
| true / false | + / − | — |
| lies / truthful | 0 / 1 | — |
| before / after | < / > | —（v8 ordering） |
| cycle | × | contradiction（v8） |
| first / last / next | ¡ / # / ^ | —（v8） |
| ready / order | % / :: | —（v8） |

其他含字母的词删除；数字、运算符、标点、空白保留（连续空白折叠）。
