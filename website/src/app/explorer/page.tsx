"use client";

import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { SessionViewer } from "@/components/explorer/session-viewer";
import { MetricSelector } from "@/components/explorer/metric-selector";
import { sampleSearchSession, sampleConvSession } from "@/lib/sample-data";

export default function ExplorerPage() {
    const [sessionType, setSessionType] = useState<"search" | "conversational">("search");
    const [selectedMetrics, setSelectedMetrics] = useState<string[]>(["jsd_action_types"]);

    const currentSession = sessionType === "search" ? sampleSearchSession : sampleConvSession;

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="space-y-2">
                <h1 className="text-3xl font-bold">Session Explorer</h1>
                <p className="text-muted-foreground">
                    Visualize and compare real vs simulated sessions. Select metrics to compute behavioral realism scores.
                </p>
            </div>

            {/* Controls */}
            <div className="flex items-center gap-4">
                <div className="space-y-1">
                    <label className="text-sm font-medium">Session Type</label>
                    <Tabs value={sessionType} onValueChange={(v) => setSessionType(v as "search" | "conversational")}>
                        <TabsList>
                            <TabsTrigger value="search" className="gap-2">
                                <span>🔍</span> Search (T)
                            </TabsTrigger>
                            <TabsTrigger value="conversational" className="gap-2">
                                <span>💬</span> Conversational (C)
                            </TabsTrigger>
                        </TabsList>
                    </Tabs>
                </div>

                <div className="space-y-1">
                    <label className="text-sm font-medium">Dataset</label>
                    <Select defaultValue="sample">
                        <SelectTrigger className="w-48">
                            <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="sample">Sample Data</SelectItem>
                            <SelectItem value="trec">TREC Session 2014</SelectItem>
                            <SelectItem value="cast">TREC CAsT 2022</SelectItem>
                        </SelectContent>
                    </Select>
                </div>
            </div>

            <div className="grid lg:grid-cols-3 gap-6">
                {/* Left Panel: Session Viewer */}
                <div className="lg:col-span-2 space-y-6">
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center justify-between">
                                <span>Session Viewer</span>
                                <Badge variant="outline">{currentSession.session_type}</Badge>
                            </CardTitle>
                            <CardDescription>
                                Interactive visualization of session events
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <SessionViewer session={currentSession} />
                        </CardContent>
                    </Card>

                    {/* Comparison View */}
                    <Card>
                        <CardHeader>
                            <CardTitle>Real vs Simulated</CardTitle>
                            <CardDescription>
                                Side-by-side comparison of session characteristics
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <div className="flex items-center gap-2">
                                        <div className="w-3 h-3 rounded-full bg-green-500" />
                                        <span className="font-medium">Real Sessions</span>
                                    </div>
                                    <div className="bg-muted/50 rounded-lg p-4 space-y-2 text-sm">
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Avg. Queries</span>
                                            <span>3.2</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Avg. Clicks</span>
                                            <span>4.8</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Avg. Duration</span>
                                            <span>142s</span>
                                        </div>
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <div className="flex items-center gap-2">
                                        <div className="w-3 h-3 rounded-full bg-blue-500" />
                                        <span className="font-medium">Simulated Sessions</span>
                                    </div>
                                    <div className="bg-muted/50 rounded-lg p-4 space-y-2 text-sm">
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Avg. Queries</span>
                                            <span>2.9</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Avg. Clicks</span>
                                            <span>5.1</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-muted-foreground">Avg. Duration</span>
                                            <span>128s</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* Right Panel: Metrics */}
                <div className="space-y-6">
                    <MetricSelector
                        selected={selectedMetrics}
                        onSelectionChange={setSelectedMetrics}
                    />

                    <Card>
                        <CardHeader>
                            <CardTitle>Results</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="space-y-3">
                                <div className="flex justify-between items-center">
                                    <span className="text-sm">JSD Action Types</span>
                                    <Badge variant="secondary">0.142</Badge>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-sm">Session Length KS</span>
                                    <Badge variant="secondary">0.089</Badge>
                                </div>
                            </div>
                            <Separator />
                            <Button className="w-full" size="sm">
                                Export Results
                            </Button>
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    );
}
