import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

interface Metric {
    name: string;
    displayName: string;
    objective: "behavior" | "evaluation" | "tester";
    granularity: "turn" | "session" | "system";
    scenarios: string[];
    description: string;
    range: string;
}

const METRICS: Metric[] = [
    {
        name: "jsd_action_types",
        displayName: "JSD Action Types",
        objective: "behavior",
        granularity: "session",
        scenarios: ["T", "C"],
        description: "Jensen-Shannon Divergence between action type distributions. Measures how similarly real and simulated sessions distribute their action types (queries, clicks, utterances, etc.).",
        range: "[0, 1] — 0 = identical, 1 = maximally different",
    },
    {
        name: "session_fd",
        displayName: "Fréchet Distance",
        objective: "behavior",
        granularity: "session",
        scenarios: ["T", "C"],
        description: "Fréchet Distance between session embedding distributions. Compares the overall 'shape' of real vs simulated session populations in embedding space.",
        range: "[0, ∞) — lower is better",
    },
    {
        name: "realism_classifier",
        displayName: "Realism Classifier",
        objective: "behavior",
        granularity: "session",
        scenarios: ["T", "C"],
        description: "Trains a classifier to distinguish real from simulated sessions. Returns 1 - AUC, so higher values indicate more realistic simulation (classifier can't distinguish).",
        range: "[0, 0.5] — 0.5 = perfect realism",
    },
    {
        name: "session_length_distribution",
        displayName: "Session Length KS",
        objective: "behavior",
        granularity: "session",
        scenarios: ["T", "C"],
        description: "Kolmogorov-Smirnov statistic comparing session length distributions.",
        range: "[0, 1] — lower is better",
    },
    {
        name: "timing_distribution",
        displayName: "Timing Distribution",
        objective: "behavior",
        granularity: "turn",
        scenarios: ["T", "C"],
        description: "KS statistic for inter-event timing distributions.",
        range: "[0, 1] — lower is better",
    },
    {
        name: "session_ndcg",
        displayName: "Session nDCG",
        objective: "evaluation",
        granularity: "session",
        scenarios: ["T"],
        description: "Session-level nDCG aggregated across all queries in the session. Uses relevance judgments from ranked items.",
        range: "[0, 1] — higher is better",
    },
    {
        name: "egu",
        displayName: "Expected Global Utility",
        objective: "evaluation",
        granularity: "session",
        scenarios: ["T"],
        description: "Models expected utility gain from a session, accounting for continuation probability and per-action utility.",
        range: "[0, ∞) — higher is better",
    },
    {
        name: "kendall_tau",
        displayName: "Kendall τ",
        objective: "tester",
        granularity: "system",
        scenarios: ["T", "C"],
        description: "Rank correlation between system rankings from two testers. Measures agreement in ordering.",
        range: "[-1, 1] — 1 = perfect agreement",
    },
    {
        name: "spearman_rho",
        displayName: "Spearman ρ",
        objective: "tester",
        granularity: "system",
        scenarios: ["T", "C"],
        description: "Spearman's rank correlation between tester rankings.",
        range: "[-1, 1] — 1 = perfect agreement",
    },
    {
        name: "rate",
        displayName: "RATE",
        objective: "tester",
        granularity: "system",
        scenarios: ["T", "C"],
        description: "RATE-style tester reliability estimation. Iteratively estimates reliability weights for each tester based on agreement with consensus.",
        range: "[0, ∞) — higher is better",
    },
];

export default function MetricsPage() {
    const behaviorMetrics = METRICS.filter((m) => m.objective === "behavior");
    const evaluationMetrics = METRICS.filter((m) => m.objective === "evaluation");
    const testerMetrics = METRICS.filter((m) => m.objective === "tester");

    return (
        <div className="space-y-10">
            <div className="space-y-2">
                <h1 className="text-3xl font-bold">Metrics Reference</h1>
                <p className="text-muted-foreground">
                    Complete documentation of all available metrics organized by objective.
                </p>
            </div>

            <MetricSection
                title="Behavioral Realism"
                description="Measure how well simulators replicate real user behavior"
                metrics={behaviorMetrics}
                color="indigo"
            />

            <Separator />

            <MetricSection
                title="System Effectiveness"
                description="Evaluate system performance with session-aware metrics"
                metrics={evaluationMetrics}
                color="green"
            />

            <Separator />

            <MetricSection
                title="Tester Reliability"
                description="Assess agreement between testers and estimate reliability"
                metrics={testerMetrics}
                color="orange"
            />
        </div>
    );
}

function MetricSection({
    title,
    description,
    metrics,
    color,
}: {
    title: string;
    description: string;
    metrics: Metric[];
    color: "indigo" | "green" | "orange";
}) {
    const colorClasses = {
        indigo: "from-indigo-500/10 to-purple-500/10 border-indigo-500/30",
        green: "from-green-500/10 to-emerald-500/10 border-green-500/30",
        orange: "from-orange-500/10 to-amber-500/10 border-orange-500/30",
    };

    return (
        <section className="space-y-4">
            <div>
                <h2 className="text-2xl font-semibold">{title}</h2>
                <p className="text-muted-foreground">{description}</p>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
                {metrics.map((metric) => (
                    <Card
                        key={metric.name}
                        className={`bg-gradient-to-br ${colorClasses[color]}`}
                    >
                        <CardHeader>
                            <CardTitle className="flex items-center justify-between">
                                <span>{metric.displayName}</span>
                                <div className="flex gap-1">
                                    {metric.scenarios.map((s) => (
                                        <Badge key={s} variant="outline" className="text-xs">
                                            {s}
                                        </Badge>
                                    ))}
                                </div>
                            </CardTitle>
                            <CardDescription>
                                <code className="text-xs bg-muted/50 px-1 py-0.5 rounded">
                                    {metric.name}
                                </code>
                            </CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-3">
                            <p className="text-sm">{metric.description}</p>
                            <div className="flex items-center gap-2 text-xs">
                                <Badge variant="secondary">{metric.granularity}</Badge>
                                <span className="text-muted-foreground">Range: {metric.range}</span>
                            </div>
                        </CardContent>
                    </Card>
                ))}
            </div>
        </section>
    );
}
