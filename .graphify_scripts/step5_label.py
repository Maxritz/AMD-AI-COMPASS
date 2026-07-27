import json
from pathlib import Path
from graphify.build import build_from_json
from graphify.cluster import score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate

extraction = json.loads(Path('graphify-out/.graphify_extract.json').read_text(encoding='utf-8'))
detection  = json.loads(Path('graphify-out/.graphify_detect.json').read_text(encoding='utf-8'))
analysis   = json.loads(Path('graphify-out/.graphify_analysis.json').read_text(encoding='utf-8'))

G = build_from_json(extraction, root='./', directed=False)
communities = {int(k): v for k, v in analysis['communities'].items()}
cohesion = {int(k): v for k, v in analysis['cohesion'].items()}
tokens = {'input': extraction.get('input_tokens', 0), 'output': extraction.get('output_tokens', 0)}

# See nodes per community to label them
for cid, nodes in sorted(communities.items(), key=lambda x: -len(x[1])):
    sample = []
    for nid in nodes[:8]:
        node_data = next((n for n in extraction['nodes'] if n['id'] == nid), None)
        if node_data:
            label = node_data.get('label', node_data.get('name', nid))
            sample.append(str(label)[:50])
    print(f'C{cid} ({len(nodes)} nodes): {sample}')

# CHOOSE LABELS based on the output above
labels = {
    0:  "Plugin System & CLI Framework",
    1:  "GPU Metrics & Performance Data",
    2:  "Kernel Tracer & DLL Injection",
    3:  "Types & Data Pipeline",
    4:  "GPU Architecture Detection",
    5:  "GPUPerfAPI Counter Adapter",
    6:  "Analysis Engine & Reports",
    7:  "Rocprof Output Formatter",
    8:  "Build Configuration",
    9:  "ADLX GPU Metrics Poller",
    10: "GEAK Plugin",
    11: "Hyperloom Plugin",
    12: "Magpie Plugin",
    13: "intellikit Plugin",
    14: "Benchmark Runner",
    15: "Git & Setup Scripts",
}

questions = suggest_questions(G, communities, labels)
report = generate(G, communities, cohesion, labels, analysis['gods'], analysis['surprises'], detection, tokens, './', suggested_questions=questions)
Path('graphify-out/GRAPH_REPORT.md').write_text(report, encoding='utf-8')
Path('graphify-out/.graphify_labels.json').write_text(json.dumps({str(k): v for k, v in labels.items()}, ensure_ascii=False), encoding='utf-8')
print('Report updated with community labels')
print()
print(f'God Nodes: {analysis["gods"][:5]}')
print(f'Surprising Connections: {analysis["surprises"][:3]}')
