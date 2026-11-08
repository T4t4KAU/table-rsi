"""Build a self-contained experimental report from completed recorded runs."""

import csv
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compare_runs import compare

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
study = root / "study-v4"
result = json.loads((study / "results.json").read_text())
aliases = result["aliases"]
labels = list(result["summary"])
comparison = compare([study / "test" / aliases[k] for k in labels])
(out / "paired_comparison.json").write_text(json.dumps(comparison, indent=2))
rows = []
for label, ci in zip(labels, comparison["runs"]):
    s = result["summary"][label]
    rows.append(
        {
            "method": label,
            "prediction_source": aliases[label],
            **s,
            "ci_low": ci["cluster_bootstrap_95_ci"][0],
            "ci_high": ci["cluster_bootstrap_95_ci"][1],
            "gain_vs_direct": ci["gain_vs_first"],
            "gain_ci_low": ci["paired_gain_95_ci"][0],
            "gain_ci_high": ci["paired_gain_95_ci"][1],
        }
    )
with (out / "results.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    }
)
fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
values = [r["turn_macro_coverage"] * 100 for r in rows]
err = [
    [100 * (r["turn_macro_coverage"] - r["ci_low"]) for r in rows],
    [100 * (r["ci_high"] - r["turn_macro_coverage"]) for r in rows],
]
ax.bar(
    labels,
    values,
    color=["#9aa4b2", "#6b819f", "#2e5d96", "#207f82", "#159c90"],
    width=0.58,
    yerr=err,
    capsize=4,
    error_kw={"linewidth": 1},
)
for i, v in enumerate(values):
    ax.text(
        i,
        rows[i]["ci_high"] * 100 + 1.5,
        f"{v:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
    )
ax.set_xlabel(
    "47 turns / 12 groups; fixed, L1 and L2 share predictions",
    fontsize=8,
    color="#596b7d",
    labelpad=10,
)
ax.set_ylim(0, 105)
ax.set_ylabel("Reference-point coverage (%)")
ax.set_title("TableRSI preliminary held-out evaluation", loc="left", fontweight="bold")
ax.grid(axis="y", alpha=0.18)
ax.set_axisbelow(True)
fig.savefig(out / "coverage.svg")
fig.savefig(out / "coverage.png", dpi=200)
plt.close(fig)


def table(headers, data):
    return (
        "<table><thead><tr>"
        + "".join("<th>" + html.escape(str(x)) + "</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join("<td>" + html.escape(str(x)) + "</td>" for x in row)
            + "</tr>"
            for row in data
        )
        + "</tbody></table>"
    )


main_table = table(
    [
        "方法",
        "要点覆盖率",
        "95% 分组区间",
        "完整覆盖轮次",
        "输入 / 输出 tokens",
        "实际预测来源",
    ],
    [
        [
            r["method"],
            f"{r['turn_macro_coverage']:.2%}",
            f"{r['ci_low']:.1%}–{r['ci_high']:.1%}",
            f"{r['fully_covered_turn_rate']:.2%}",
            f"{r['local_input_tokens']:,} / {r['local_output_tokens']:,}",
            r["prediction_source"],
        ]
        for r in rows
    ],
)
audit = []
cost = []
for level in ["L1", "L2"]:
    for p in sorted((study / level).glob("attempt*/decision.json")):
        d = json.loads(p.read_text())
        g = d.get("gate", {})
        audit.append(
            [
                level,
                p.parent.name,
                d.get("operator", "—"),
                str(g.get("accept", False)),
                f"{g['gain']:+.3f}" if g.get("gain") is not None else "—",
                f"{g['lower_95_one_sided']:+.3f}" if "lower_95_one_sided" in g else "—",
                g.get("reason", d.get("reason", "")),
            ]
        )
    summaries = [
        json.loads(p.read_text())
        for p in (study / level).glob("attempt*/*/summary.json")
    ]
    cost.append(
        [
            level,
            len(summaries),
            sum(s["turns"] for s in summaries),
            sum(s["local_input_tokens"] for s in summaries),
            sum(s["local_output_tokens"] for s in summaries),
            sum(s["judge_input_tokens"] for s in summaries),
            sum(s["judge_output_tokens"] for s in summaries),
        ]
    )
selection = json.loads((study / "model_selection.json").read_text())
devtable = table(
    ["冻结模型", "开发轮次", "开发覆盖率", "评分失败", "未知模型用量"],
    [
        [
            m,
            r["turns"],
            f"{r['turn_macro_coverage']:.2%}",
            r["judge_failures"],
            r["unknown_model_usage_calls"],
        ]
        for m, r in selection["reports"].items()
    ],
)
errors = sum(
    result["summary"][k]["judge_failures"]
    + result["summary"][k]["unknown_model_usage_calls"]
    for k in set(aliases.values())
)
accepted = sum(row[3] == "True" for row in audit)
headline = f"{result['selected_model']} · 12 个测试表格组 · 47 轮多轮问题"
orders = {}
for level in ("L1", "L2"):
    orders[level] = [
        json.loads(p.read_text())
        for p in sorted((study / level).glob("attempt*/proposal-order.json"))
    ]
changed = any(
    set(a["operators"]) != set(b["operators"])
    for a, b in zip(orders["L1"], orders["L2"])
)
if aliases["L1"] == aliases["L2"]:
    finding = "L1 与 L2 的最终策略和记忆相同，终测复用同一份预测。此次实验不能据此宣称 RSI 带来准确率提升。"
else:
    delta = (
        result["summary"]["L2"]["turn_macro_coverage"]
        - result["summary"]["L1"]["turn_macro_coverage"]
    )
    finding = f"L2 相对 L1 的终测覆盖率变化为 {delta:+.2%}。样本量有限，需结合分组区间与更多任务判断，不能把单次差异当作普遍收益。"
fixed = result["summary"]["fixed"]
retry = result["summary"]["retry"]
finding += f" 固定搜索覆盖率为 {fixed['turn_macro_coverage']:.2%}，SQL 重试为 {retry['turn_macro_coverage']:.2%}；固定搜索使用了后者 {(fixed['local_input_tokens'] + fixed['local_output_tokens']) / (retry['local_input_tokens'] + retry['local_output_tokens']):.2f} 倍的本地 tokens，本轮主指标与成本未显示优势。"
if not changed:
    finding += (
        " 两层实际评估的候选集合没有发生变化，因此本次尚不足以隔离元策略的因果收益。"
    )
body = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TableRSI 实现与实验记录</title><style>
:root{{--ink:#17283a;--blue:#2e5d96;--muted:#596b7d;--line:#dce5ed}}*{{box-sizing:border-box}}body{{margin:0;background:#f2f5f8;color:var(--ink);font-family:system-ui,"Noto Sans CJK SC",sans-serif;line-height:1.7}}main{{max-width:1080px;margin:36px auto;background:white;padding:48px;border-radius:16px;box-shadow:0 12px 50px #213b5b0c}}.eyebrow{{color:var(--blue);letter-spacing:.12em;font-size:12px;font-weight:700}}h1{{font-size:32px;line-height:1.25;margin:12px 0}}h2{{font-size:21px;margin-top:36px;border-top:1px solid var(--line);padding-top:24px}}p{{max-width:940px}}.sub{{color:var(--muted)}}.callout{{background:#edf4fa;border-left:4px solid var(--blue);padding:16px 20px;margin:24px 0}}table{{border-collapse:collapse;width:100%;font-size:13px;line-height:1.5}}th{{text-align:left;background:#eef3f8}}td,th{{padding:10px 9px;border-bottom:1px solid var(--line);vertical-align:top}}.scroll{{overflow-x:auto}}figure{{margin:24px 0}}figure img{{width:100%;max-width:860px}}figcaption,small{{color:var(--muted);font-size:12px}}a{{color:var(--blue)}}code{{background:#edf2f7;padding:2px 5px;font-size:12px}}@media(max-width:700px){{main{{margin:0;padding:24px;border-radius:0}}h1{{font-size:26px}}}}@media print{{body{{background:white}}main{{box-shadow:none;margin:0;padding:0}}h2{{break-after:avoid}}tr{{break-inside:avoid}}}}
</style><main><div class="eyebrow">TABLERSI · RECORDED EXPERIMENT</div><h1>冻结模型的表格推理与跨任务改进</h1><p class="sub">{headline}</p>
<div class="callout"><strong>结果边界：</strong>这是可复现原型的初步实验，不是完整 benchmark 排名。指标是 LLM 按参考要点评定的覆盖率，不是人工确认的准确率。{html.escape(finding)}</div>
<h2>独立测试结果</h2><p>先在开发集选择模型，再以每段回放对话的首个问题进行 proposal / fresh audit，固定策略后在完整多轮对话上进行终测。表格组不跨分区。所有方法使用同一模型与资源上限；实际调用量不同。终测异常计数：{errors}。</p><div class="scroll">{main_table}</div><figure><img src="coverage.svg" alt="终测要点覆盖率及95%分组bootstrap区间"><figcaption>按表格内容组进行 4,000 次 bootstrap；区间为描述性 percentile 区间，未做多重比较校正，且不涵盖生成与评分器的随机波动。同源预测标签的分数完全相同，避免重复评分噪声。</figcaption></figure>
<h2>模型选择与开发隔离</h2>{devtable}<p>两个模型均由 ModelScope 下载，使用 BF16 冻结推理。开发阶段修复了上下文超长、列坐标提示及评分分母结构问题。旧版试跑记录保留在原始日志中，不混入终测。模型选择未查看测试答案或分数。开发模型比较先于最终补丁权限修复；主试验使用修复后冻结代码，旧试跑单独保留。</p>
<h2>两层改进的实际决策</h2><p>首轮 L1/L2 的候选、任务和评分结果完全共享；L2 在完成审计后才更新下一轮算子分布。只有通过独立审计才能提交求解策略或记忆。本次共有 {accepted} 个按层计数的接受决策（共享首轮可能在两层中各计一次）。</p><div class="scroll">{table(["层级", "轮次", "算子", "接受", "平均增益", "单侧下界", "决策说明"], audit)}</div>
<h2>更新成本</h2><div class="scroll">{table(["层级", "评估批次", "回放轮数", "本地输入 tokens", "本地输出 tokens", "评分输入 tokens", "评分输出 tokens"], cost)}</div><p>以上为每个算法的逻辑成本，均计入被拒绝候选及审计。首轮物理执行仅一次。实际 API 费用需以供应商账单为准；这里不推算未经核验的金额。延迟包含共享服务排队，不能视作独占 GPU 性能。</p>
<h2>实现范围</h2><p>已实现：角色化 SQL 搜索、只读隔离执行、受限单字面量补丁、证据准入接口、版本化策略/抽象记忆候选、独立审计提交及 L2 算子重加权。SQL 由完整语句执行；尚未实现通用 DAG 节点缓存复用、自动归纳的可执行修复技能或 teacher SFT。结构权限检查不保证语义正确。</p><p>原始数据包含 6 个缺失表目录任务和 1 个含缺失问题字段的任务；抽中时不通过标准答案补造输入。分组仅覆盖精确规范化表内容去重，不保证消除所有模板重复或预训练污染。开发/审计评分依赖云 API，故此次实验不能称为完全离线自适应。</p>
<h2>复核材料</h2><p><a href="results.csv">逐方法指标 CSV</a> · <a href="paired_comparison.json">配对分组区间</a> · <a href="experiment_protocol.json">实验协议</a> · <a href="TableRSI_source.zip">代码、测试与复现脚本</a> · <a href="implementation/IMPLEMENTATION_NOTES.md">实现边界说明</a></p><p class="sub">数据：<a href="https://github.com/201983290498/TableAgentBench">TableAgentBench</a>，固定提交 f70f134f63e4a7b0c4d15b2540953b5e95388586。模型：<a href="https://modelscope.cn/models/Qwen/Qwen3-8B">Qwen3-8B</a> / <a href="https://modelscope.cn/models/Qwen/Qwen3-32B">Qwen3-32B</a>。服务与数据根目录：<code>/data1/sdnware/tablersi</code>。</p></main></html>"""
(out / "experiment_report.html").write_text(body)
print(
    json.dumps(
        {
            "report": str(out / "experiment_report.html"),
            "accepted_decisions": accepted,
            "finding": finding,
        },
        ensure_ascii=False,
    )
)
