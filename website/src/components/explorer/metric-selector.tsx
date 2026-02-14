"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface Metric {
    name: string;
    displayName: string;
    objective: "behavior" | "evaluation" | "tester";
    description: string;
}

const METRICS: Metric[] = [
    { name: "jsd_action_types", displayName: "JSD Action Types", objective: "behavior", description: "Distribution of action types" },
    { name: "session_fd", displayName: "Fréchet Distance", objective: "behavior", description: "Session embedding distance" },
    { name: "realism_classifier", displayName: "Classifier AUC", objective: "behavior", description: "Can a classifier distinguish?" },
    { name: "session_length_distribution", displayName: "Session Length KS", objective: "behavior", description: "Session length distribution" },
    { name: "session_ndcg", displayName: "Session nDCG", objective: "evaluation", description: "Session-level ranking quality" },
    { name: "egu", displayName: "Expected Global Utility", objective: "evaluation", description: "Session utility measure" },
    { name: "kendall_tau", displayName: "Kendall τ", objective: "tester", description: "Rank correlation" },
    { name: "rate", displayName: "RATE", objective: "tester", description: "Reliability estimation" },
];

interface MetricSelectorProps {
    selected: string[];
    onSelectionChange: (selected: string[]) => void;
}

export function MetricSelector({ selected, onSelectionChange }: MetricSelectorProps) {
    const toggleMetric = (name: string) => {
        if (selected.includes(name)) {
            onSelectionChange(selected.filter((m) => m !== name));
        } else {
            onSelectionChange([...selected, name]);
        }
    };

    const objectiveGroups = {
        behavior: METRICS.filter((m) => m.objective === "behavior"),
        evaluation: METRICS.filter((m) => m.objective === "evaluation"),
        tester: METRICS.filter((m) => m.objective === "tester"),
    };

    return (
        <Card>
            <CardHeader>
                <CardTitle>Select Metrics</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
                <MetricGroup
                    title="Behavioral Realism"
                    metrics={objectiveGroups.behavior}
                    selected={selected}
                    onToggle={toggleMetric}
                    color="indigo"
                />
                <MetricGroup
                    title="System Effectiveness"
                    metrics={objectiveGroups.evaluation}
                    selected={selected}
                    onToggle={toggleMetric}
                    color="green"
                />
                <MetricGroup
                    title="Tester Reliability"
                    metrics={objectiveGroups.tester}
                    selected={selected}
                    onToggle={toggleMetric}
                    color="orange"
                />
            </CardContent>
        </Card>
    );
}

function MetricGroup({
    title,
    metrics,
    selected,
    onToggle,
    color,
}: {
    title: string;
    metrics: Metric[];
    selected: string[];
    onToggle: (name: string) => void;
    color: "indigo" | "green" | "orange";
}) {
    const colorClasses = {
        indigo: "border-indigo-500/30",
        green: "border-green-500/30",
        orange: "border-orange-500/30",
    };

    return (
        <div className="space-y-2">
            <h4 className="text-sm font-medium text-muted-foreground">{title}</h4>
            <div className="space-y-1">
                {metrics.map((metric) => {
                    const isSelected = selected.includes(metric.name);
                    return (
                        <button
                            key={metric.name}
                            onClick={() => onToggle(metric.name)}
                            className={`w-full text-left p-2 rounded-md text-sm transition-colors border ${isSelected
                                    ? `bg-${color}-500/10 ${colorClasses[color]}`
                                    : "border-transparent hover:bg-muted/50"
                                }`}
                        >
                            <div className="flex items-center justify-between">
                                <span className={isSelected ? "font-medium" : ""}>{metric.displayName}</span>
                                {isSelected && <Badge variant="secondary" className="text-xs">✓</Badge>}
                            </div>
                        </button>
                    );
                })}
            </div>
        </div>
    );
}
