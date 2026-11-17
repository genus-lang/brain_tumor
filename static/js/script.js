/**
 * NeuroScan AI - Medical AI Dashboard Client Script
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

    const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png"];
    const MAX_SIZE_MB = 15;

    // Helper: format file size
    function formatBytes(bytes) {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    }

    // Display selected file with preview & metadata
    function displayFile(file) {
        if (!file) return;

        // Validation: format
        if (!ALLOWED_TYPES.includes(file.type.toLowerCase()) && 
            !file.name.match(/\.(jpg|jpeg|png)$/i)) {
            alert("Invalid file type. Please upload a standard MRI scan in JPG, JPEG, or PNG format.");
            if (fileInput) fileInput.value = "";
            return;
        }

        // Validation: size
        if (file.size > MAX_SIZE_MB * 1024 * 1024) {
            alert(`File size exceeds ${MAX_SIZE_MB}MB limit. Please upload a smaller scan.`);
            if (fileInput) fileInput.value = "";
            return;
        }

        // Revoke prior URL to prevent memory leaks
        if (previewURL) {
            URL.revokeObjectURL(previewURL);
        }

        previewURL = URL.createObjectURL(file);
        previewImage.src = previewURL;

        // Image dimensions on load
        previewImage.onload = () => {
            if (previewDimensions) {
                previewDimensions.textContent = `${previewImage.naturalWidth} × ${previewImage.naturalHeight} px`;
            }
        };

        if (previewFileName) {
            previewFileName.textContent = file.name;
        }

        if (previewSize) {
            previewSize.textContent = formatBytes(file.size);
        }

        if (previewWrapper) {
            previewWrapper.style.display = "block";
        }

        if (dropArea) {
            dropArea.style.borderColor = "#2563eb";
        }
    }

    // File input change handler
    if (fileInput) {
        fileInput.addEventListener("change", function () {
            if (this.files && this.files[0]) {
                displayFile(this.files[0]);
            }
        });
    }

    // Drag-and-drop support
    if (dropArea) {
        ["dragenter", "dragover"].forEach((eventName) => {
            dropArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropArea.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            dropArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropArea.classList.remove("dragover");
            });
        });

        dropArea.addEventListener("drop", (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                fileInput.files = files;
                displayFile(files[0]);
            }
        });
    }

    // Remove selected file handler
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

    // Loading spinner on form submission
    if (uploadForm) {
        uploadForm.addEventListener("submit", (e) => {
            if (!fileInput || !fileInput.files || !fileInput.files.length) {
                e.preventDefault();
                alert("Please select or drop an MRI scan before analyzing.");
                return;
            }

            if (analyzeBtn) {
                analyzeBtn.disabled = true;
                analyzeBtn.innerHTML = `
                    <i class="fa-solid fa-spinner fa-spin"></i>
                    <span>Analyzing MRI Tensor...</span>
                `;
            }
        });
    }

    // Animate confidence bar and differential bars if present
    const confidenceBar = document.getElementById("confidenceBar");
    if (confidenceBar) {
        const rawConfidence = confidenceBar.getAttribute("data-confidence") || "0";
        const val = parseFloat(rawConfidence.replace("%", "").trim());
        const safeVal = Math.max(0, Math.min(100, isNaN(val) ? 0 : val));

        setTimeout(() => {
            confidenceBar.style.width = safeVal + "%";
        }, 150);

        // Smooth scroll to results
        const resultSection = document.getElementById("results");
        if (resultSection) {
            resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    }

    // Animate differential breakdown micro-bars
    document.querySelectorAll(".diff-fill").forEach((bar) => {
        const targetVal = bar.getAttribute("data-value") || "0";
        const numVal = parseFloat(targetVal);
        setTimeout(() => {
            bar.style.width = Math.max(0, Math.min(100, isNaN(numVal) ? 0 : numVal)) + "%";
        }, 300);
    });
});

