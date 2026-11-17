/**
 * heatmap.js - NeuroScan AI Grad-CAM Multi-Modal Viewer
 * Coordinates smooth transitions between Original MRI, Pure Heatmap, and Overlay.
 */

document.addEventListener("DOMContentLoaded", () => {
    const viewerToggle = document.getElementById("viewerToggle");
    const viewerImg = document.getElementById("activeViewerImg");
    const viewerCaption = document.getElementById("viewerCaption");

    if (!viewerToggle || !viewerImg) return;

    const buttons = viewerToggle.querySelectorAll(".toggle-btn");

    const captions = {
        overlay: '<i class="fa-solid fa-layer-group me-1"></i> Showing: <strong>Grad-CAM Overlay</strong> (Alpha blend: 0.42 heatmap + 0.58 MRI)',
        heatmap: '<i class="fa-solid fa-fire me-1"></i> Showing: <strong>Pure Grad-CAM Activation Heatmap</strong> (JET spectrum colormap)',
        original: '<i class="fa-regular fa-image me-1"></i> Showing: <strong>Original Unaltered MRI Scan</strong> (Axial view)'
    };

    buttons.forEach(btn => {
        btn.addEventListener("click", () => {
            const viewMode = btn.getAttribute("data-view");
            if (!viewMode) return;

            // Update active state
            buttons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            // Determine target image URL
            let targetSrc = "";
            if (viewMode === "overlay") {
                targetSrc = viewerImg.getAttribute("data-overlay");
            } else if (viewMode === "heatmap") {
                targetSrc = viewerImg.getAttribute("data-heatmap");
            } else if (viewMode === "original") {
                targetSrc = viewerImg.getAttribute("data-original");
            }

            if (targetSrc) {
                // Smooth fade effect
                viewerImg.style.opacity = "0.3";
                setTimeout(() => {
                    viewerImg.src = targetSrc;
                    viewerImg.style.opacity = "1";
                }, 120);
            }

            // Update caption
            if (viewerCaption && captions[viewMode]) {
                viewerCaption.innerHTML = captions[viewMode];
            }
        });
    });
});

