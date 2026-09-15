
fetch("http://localhost:5000/detect", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ url: window.location.href })
})
.then(res => res.json())
.then(data => {
const api = typeof browser !== "undefined" ? browser : api;

api.runtime.sendMessage({ action: "goGoogle" });
  console.log("DATA:", data);
// 🔵 Trusted websites list
const trustedSites = [
  "github.com",
  "google.com",
  "youtube.com",
  "localhost",
  "elearning.mti.edu.eg",
  "angham.com",
];

// 🔵 Get current hostname
const hostname = window.location.hostname;

// 🔵 Check if trusted
const isTrusted = trustedSites.some(site =>
  hostname === site || hostname.endsWith("." + site)
);

if (isTrusted) {
  console.log("✅ Trusted site — skipping detection");

  // OPTIONAL UI
  const trustedBox = document.createElement("div");

  trustedBox.style.position = "fixed";
  trustedBox.style.top = "20px";
  trustedBox.style.right = "20px";
  trustedBox.style.padding = "10px";
  trustedBox.style.background = "#1890ff";
  trustedBox.style.color = "white";
  trustedBox.style.zIndex = "999999";

  trustedBox.innerText = "🔵 Trusted Website";

  document.body.appendChild(trustedBox);

  setTimeout(() => trustedBox.remove(), 3000);

  // ❗ VERY IMPORTANT: STOP EXECUTION
  return;
}
  // 🔥 simple logic
  const isPhishing = data.prediction == 1 || data.prediction === "phishing";

  if (isPhishing) {

  const overlay = document.createElement("div");

  overlay.style.position = "fixed";
  overlay.style.top = "0";
  overlay.style.left = "0";
  overlay.style.width = "100%";
  overlay.style.height = "100%";
  overlay.style.background = "#fff";
  overlay.style.zIndex = "999999";
  overlay.style.display = "flex";
  overlay.style.flexDirection = "column";
  overlay.style.alignItems = "center";
  overlay.style.justifyContent = "center";
  overlay.style.fontFamily = "Segoe UI";

  overlay.innerHTML = `
    <div style="text-align:center; max-width:500px;">
      
      <h1 style="color:#d93025;">⚠️ Dangerous site ahead</h1>
      
      <p style="font-size:16px; color:#333;">
        This website may try to steal your personal information.
      </p>

      <div style="margin-top:25px;">
        
        <button id="goBack" style="
          padding:12px 20px;
          background:#d93025;
          color:white;
          border:none;
          border-radius:8px;
          cursor:pointer;
          margin-right:10px;
        ">
          Go Back
        </button>

        <button id="ignore" style="
          padding:12px 20px;
          background:#eee;
          border:none;
          border-radius:8px;
          cursor:pointer;
        ">
          Ignore (Unsafe)
        </button>

      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  // 🔴 Go back
  document.getElementById("goBack").onclick = () => {
  window.location.href = "https://www.google.com";

  };

  // ⚠️ Ignore warning
  document.getElementById("ignore").onclick = () => {
    overlay.remove();
  };


  } else {

    const safeBox = document.createElement("div");

    safeBox.style.position = "fixed";
    safeBox.style.top = "20px";
    safeBox.style.right = "20px";
    safeBox.style.padding = "10px";
    safeBox.style.background = "green";
    safeBox.style.color = "white";
    safeBox.style.zIndex = "999999";

    safeBox.innerText = "✅ SAFE";

    document.body.appendChild(safeBox);

    setTimeout(() => safeBox.remove(), 3000);
  }

});