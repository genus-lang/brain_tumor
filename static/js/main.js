/**
 * main.js - NeuroScan AI Core Frontend Logic
 * Manages file uploads, drag-and-drop, client validation, and UI state.
 */

document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("fileInput");
    const dropArea = document.getElementById("dropArea");
    const previewWrapper = document.getElementById("previewWrapper");
    const previewImage = document.getElementById("previewImage");
    const previewFileName = document.getElementById("previewFileName");
    const previewDimensions = document.getElementById("previewDimensions");
    const previewSize = document.getElementById("previewSize");
    const removeBtn = document.getElementById("removeBtn");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const uploadForm = document.getElementById("uploadForm");

    let previewURL = null;
    const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"];
    const MAX_SIZE_MB = 16;

    function formatBytes(bytes) {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    }

    function validateAndDisplay(file) {
        if (!file) return;

        const fileName = file.name.toLowerCase();
        const isValidExt = ALLOWED_EXTENSIONS.some(ext => fileName.endsWith(ext));

        if (!isValidExt) {
            alert("Unsupported format. Please upload an MRI scan in JPG, JPEG, or PNG format.");
            if (fileInput) fileInput.value = "";
            return;
        }

        if (file.size > MAX_SIZE_MB * 1024 * 1024) {
            alert(`File size exceeds ${MAX_SIZE_MB}MB limit. Please upload a compressed scan.`);
            if (fileInput) fileInput.value = "";
            return;
        }

        if (previewURL) {
            URL.revokeObjectURL(previewURL);
        }

        previewURL = URL.createObjectURL(file);
        if (previewImage) {
            previewImage.src = previewURL;
            previewImage.onload = () => {
                if (previewDimensions) {
                    previewDimensions.textContent = `${previewImage.naturalWidth} × ${previewImage.naturalHeight} px`;
                }
            };
        }

        if (previewFileName) previewFileName.textContent = file.name;
        if (previewSize) previewSize.textContent = formatBytes(file.size);
        if (previewWrapper) previewWrapper.style.display = "block";
        if (dropArea) dropArea.style.borderColor = "#2563eb";
    }

    if (fileInput) {
        fileInput.addEventListener("change", function () {
            if (this.files && this.files[0]) {
                validateAndDisplay(this.files[0]);
            }
        });
    }

    if (dropArea) {
        ["dragenter", "dragover"].forEach(evt => {
            dropArea.addEventListener(evt, e => {
                e.preventDefault();
                e.stopPropagation();
                dropArea.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach(evt => {
            dropArea.addEventListener(evt, e => {
                e.preventDefault();
                e.stopPropagation();
                dropArea.classList.remove("dragover");
            });
        });

        dropArea.addEventListener("drop", e => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                fileInput.files = files;
                validateAndDisplay(files[0]);
            }
        });
    }

    if (removeBtn) {
        removeBtn.addEventListener("click", () => {
            if (fileInput) fileInput.value = "";
            if (previewWrapper) previewWrapper.style.display = "none";
            if (dropArea) dropArea.style.borderColor = "#cbd5e1";
            if (previewURL) {
                URL.revokeObjectURL(previewURL);
                previewURL = null;
            }
        });
    }

    if (uploadForm) {
        uploadForm.addEventListener("submit", e => {
            if (!fileInput || !fileInput.files || !fileInput.files.length) {
                e.preventDefault();
                alert("Please select or drop an MRI scan before proceeding.");
                return;
            }

            if (analyzeBtn) {
                analyzeBtn.disabled = true;
                analyzeBtn.innerHTML = `
                    <i class="fa-solid fa-spinner fa-spin"></i>
                    <span>Analyzing Scan &amp; Generating Grad-CAM...</span>
                `;
            }
        });
    }

    // Animate progress bar if on analysis page
    const confidenceBar = document.getElementById("confidenceBar");
    if (confidenceBar) {
        const raw = confidenceBar.getAttribute("data-confidence") || "0";
        const val = parseFloat(raw.replace("%", "").trim());
        const safeVal = Math.max(0, Math.min(100, isNaN(val) ? 0 : val));
        setTimeout(() => {
            confidenceBar.style.width = safeVal + "%";
        }, 150);

        const resultsSection = document.getElementById("results");
        if (resultsSection) {
            resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    }

    // Animate differential micro-bars
    document.querySelectorAll(".diff-fill").forEach(bar => {
        const targetVal = parseFloat(bar.getAttribute("data-value") || "0");
        setTimeout(() => {
            bar.style.width = Math.max(0, Math.min(100, isNaN(targetVal) ? 0 : targetVal)) + "%";
        }, 250);
    });
});

