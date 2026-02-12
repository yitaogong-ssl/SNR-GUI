const copyButton = document.querySelector(".copy-btn");

if (copyButton) {
  copyButton.addEventListener("click", async () => {
    const targetId = copyButton.getAttribute("data-copy-target");
    const codeEl = document.getElementById(targetId);
    if (!codeEl) return;
    const text = codeEl.textContent || "";

    try {
      await navigator.clipboard.writeText(text);
      copyButton.textContent = "Copied";
      setTimeout(() => {
        copyButton.textContent = "Copy Command";
      }, 1200);
    } catch (err) {
      copyButton.textContent = "Copy failed";
      setTimeout(() => {
        copyButton.textContent = "Copy Command";
      }, 1200);
    }
  });
}
