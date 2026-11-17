/**
 * charts.js - NeuroScan AI Interactive Visualizations
 * Renders Chart.js probability distribution, accuracy, and loss curves.
 */

document.addEventListener("DOMContentLoaded", () => {
    // 1. PROBABILITY DISTRIBUTION BAR CHART
    const probCanvas = document.getElementById("probabilityChart");
    if (probCanvas && typeof Chart !== "undefined") {
        try {
            const rawLabels = probCanvas.getAttribute("data-labels");
            const rawValues = probCanvas.getAttribute("data-values");
            const rawColors = probCanvas.getAttribute("data-colors");

            const labels = rawLabels ? JSON.parse(rawLabels) : ["Pituitary", "Glioma", "No Tumor", "Meningioma"];
            const values = rawValues ? JSON.parse(rawValues) : [0, 0, 0, 0];
            const colors = rawColors ? JSON.parse(rawColors) : ["#7c3aed", "#dc2626", "#16a34a", "#d97706"];

            new Chart(probCanvas, {
                type: "bar",
                data: {
                    labels: labels,
                    datasets: [{
                        label: "Model Probability (%)",
                        data: values,
                        backgroundColor: colors,
                        borderRadius: 8,
                        borderSkipped: false,
                        barThickness: 34
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: function (context) {
                                    return ` Confidence: ${context.parsed.y.toFixed(2)}%`;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            ticks: {
                                callback: val => val + "%",
                                font: { family: "'DM Sans', sans-serif", size: 12 }
                            },
                            grid: { color: "#f1f5f9" }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { font: { family: "'DM Sans', sans-serif", weight: 700 } }
                        }
                    }
                }
            });
        } catch (err) {
            console.error("Failed to render probability chart:", err);
        }
    }

    // 2. TRAINING ACCURACY LINE CHART
    const accCanvas = document.getElementById("accuracyChart");
    if (accCanvas && typeof Chart !== "undefined") {
        try {
            const epochs = JSON.parse(accCanvas.getAttribute("data-epochs") || "[]");
            const trainAcc = JSON.parse(accCanvas.getAttribute("data-train-acc") || "[]").map(v => (v * 100).toFixed(1));
            const valAcc = JSON.parse(accCanvas.getAttribute("data-val-acc") || "[]").map(v => (v * 100).toFixed(1));

            new Chart(accCanvas, {
                type: "line",
                data: {
                    labels: epochs.map(e => `Epoch ${e}`),
                    datasets: [
                        {
                            label: "Training Accuracy",
                            data: trainAcc,
                            borderColor: "#2563eb",
                            backgroundColor: "rgba(37, 99, 235, 0.08)",
                            fill: true,
                            tension: 0.35,
                            borderWidth: 2.5,
                            pointRadius: 4,
                            pointHoverRadius: 6
                        },
                        {
                            label: "Validation Accuracy",
                            data: valAcc,
                            borderColor: "#16a34a",
                            backgroundColor: "transparent",
                            borderDash: [5, 5],
                            tension: 0.35,
                            borderWidth: 2.5,
                            pointRadius: 4,
                            pointHoverRadius: 6
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y}%`
                            }
                        }
                    },
                    scales: {
                        y: {
                            min: 60,
                            max: 100,
                            ticks: { callback: v => v + "%" },
                            grid: { color: "#f1f5f9" }
                        },
                        x: {
                            grid: { color: "#f8fafc" },
                            ticks: { font: { size: 11 } }
                        }
                    }
                }
            });
        } catch (err) {
            console.error("Failed to render accuracy chart:", err);
        }
    }

    // 3. TRAINING LOSS LINE CHART
    const lossCanvas = document.getElementById("lossChart");
    if (lossCanvas && typeof Chart !== "undefined") {
        try {
            const epochs = JSON.parse(lossCanvas.getAttribute("data-epochs") || "[]");
            const trainLoss = JSON.parse(lossCanvas.getAttribute("data-train-loss") || "[]");
            const valLoss = JSON.parse(lossCanvas.getAttribute("data-val-loss") || "[]");

            new Chart(lossCanvas, {
                type: "line",
                data: {
                    labels: epochs.map(e => `Epoch ${e}`),
                    datasets: [
                        {
                            label: "Training Loss",
                            data: trainLoss,
                            borderColor: "#dc2626",
                            backgroundColor: "rgba(220, 38, 38, 0.08)",
                            fill: true,
                            tension: 0.35,
                            borderWidth: 2.5,
                            pointRadius: 4,
                            pointHoverRadius: 6
                        },
                        {
                            label: "Validation Loss",
                            data: valLoss,
                            borderColor: "#d97706",
                            backgroundColor: "transparent",
                            borderDash: [5, 5],
                            tension: 0.35,
                            borderWidth: 2.5,
                            pointRadius: 4,
                            pointHoverRadius: 6
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(3)}`
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: { color: "#f1f5f9" }
                        },
                        x: {
                            grid: { color: "#f8fafc" },
                            ticks: { font: { size: 11 } }
                        }
                    }
                }
            });
        } catch (err) {
            console.error("Failed to render loss chart:", err);
        }
    }
});

