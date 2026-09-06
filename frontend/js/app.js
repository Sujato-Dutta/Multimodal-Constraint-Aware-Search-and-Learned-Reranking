/**
 * ConstraintSearch Frontend Application Logic.
 * Handles tab navigation, theme switching, image upload, live search,
 * pipeline animations, and benchmark metrics rendering.
 */

// Sample fallback mock results matching assets/mock_image.png exactly
const INITIAL_DEMO_RESULTS = [
  {
    product_id: "ADI-00001",
    name: "Ultraboost Light Running Shoes",
    category: "Footwear",
    subcategory: "Running Shoes",
    color: "Black",
    color_details: "Black / Core Black",
    selling_price: 110.00,
    original_price: 180.00,
    brand: "Adidas",
    rating: 4.8,
    image_url: "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=600&q=80",
    similarity_score: 0.962,
    rerank_score: 0.96,
    initial_rank: 2,
    final_rank: 1,
    constraint_status: { all_satisfied: true, category_ok: true, color_ok: true, price_ok: true }
  },
  {
    product_id: "ADI-00002",
    name: "Duramo SL 2 Running Shoes",
    category: "Footwear",
    subcategory: "Running Shoes",
    color: "Black",
    color_details: "Black / Carbon",
    selling_price: 70.00,
    original_price: 80.00,
    brand: "Adidas",
    rating: 4.6,
    image_url: "https://images.unsplash.com/photo-1584735935682-2f2b69dff9d2?auto=format&fit=crop&w=600&q=80",
    similarity_score: 0.915,
    rerank_score: 0.93,
    initial_rank: 4,
    final_rank: 2,
    constraint_status: { all_satisfied: true, category_ok: true, color_ok: true, price_ok: true }
  },
  {
    product_id: "ADI-00003",
    name: "Galaxy 6 Running Shoes",
    category: "Footwear",
    subcategory: "Running Shoes",
    color: "Black",
    color_details: "Black / Core Black",
    selling_price: 65.00,
    original_price: 75.00,
    brand: "Adidas",
    rating: 4.5,
    image_url: "https://images.unsplash.com/photo-1608231387042-66d1773070a5?auto=format&fit=crop&w=600&q=80",
    similarity_score: 0.895,
    rerank_score: 0.91,
    initial_rank: 5,
    final_rank: 3,
    constraint_status: { all_satisfied: true, category_ok: true, color_ok: true, price_ok: true }
  },
  {
    product_id: "ADI-00004",
    name: "Supernova Rise Running Shoes",
    category: "Footwear",
    subcategory: "Running Shoes",
    color: "Black",
    color_details: "Black / Silver Metallic",
    selling_price: 120.00,
    original_price: 140.00,
    brand: "Adidas",
    rating: 4.7,
    image_url: "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?auto=format&fit=crop&w=600&q=80",
    similarity_score: 0.880,
    rerank_score: 0.91,
    initial_rank: 6,
    final_rank: 4,
    constraint_status: { all_satisfied: true, category_ok: true, color_ok: true, price_ok: true }
  },
  {
    product_id: "ADI-00005",
    name: "Pureboost 23 Running Shoes",
    category: "Footwear",
    subcategory: "Running Shoes",
    color: "Black",
    color_details: "Black / Core Black",
    selling_price: 100.00,
    original_price: 130.00,
    brand: "Adidas",
    rating: 4.6,
    image_url: "https://images.unsplash.com/photo-1579338559194-a162d19bf842?auto=format&fit=crop&w=600&q=80",
    similarity_score: 0.872,
    rerank_score: 0.87,
    initial_rank: 7,
    final_rank: 5,
    constraint_status: { all_satisfied: true, category_ok: true, color_ok: true, price_ok: true }
  },
  {
    product_id: "ADI-00006",
    name: "Runfalcon 5 Running Shoes",
    category: "Footwear",
    subcategory: "Running Shoes",
    color: "Black",
    color_details: "Black / Grey",
    selling_price: 55.00,
    original_price: 65.00,
    brand: "Adidas",
    rating: 4.4,
    image_url: "https://images.unsplash.com/photo-1582588678413-dbf45f4823e9?auto=format&fit=crop&w=600&q=80",
    similarity_score: 0.841,
    rerank_score: 0.84,
    initial_rank: 8,
    final_rank: 6,
    constraint_status: { all_satisfied: true, category_ok: true, color_ok: true, price_ok: true }
  },
  {
    product_id: "ADI-00007",
    name: "Adizero SL Running Shoes",
    category: "Footwear",
    subcategory: "Running Shoes",
    color: "Black",
    color_details: "Black / Carbon",
    selling_price: 110.00,
    original_price: 120.00,
    brand: "Adidas",
    rating: 4.7,
    image_url: "https://images.unsplash.com/photo-1606107557195-0e29a4b5b4aa?auto=format&fit=crop&w=600&q=80",
    similarity_score: 0.835,
    rerank_score: 0.83,
    initial_rank: 9,
    final_rank: 7,
    constraint_status: { all_satisfied: true, category_ok: true, color_ok: true, price_ok: true }
  },
  {
    product_id: "ADI-00008",
    name: "Response Super 3.0 Running Shoes",
    category: "Footwear",
    subcategory: "Running Shoes",
    color: "Black",
    color_details: "Black / Grey Six",
    selling_price: 75.00,
    original_price: 90.00,
    brand: "Adidas",
    rating: 4.5,
    image_url: "https://images.unsplash.com/photo-1587563871167-1ee9c731aefb?auto=format&fit=crop&w=600&q=80",
    similarity_score: 0.820,
    rerank_score: 0.82,
    initial_rank: 10,
    final_rank: 8,
    constraint_status: { all_satisfied: true, category_ok: true, color_ok: true, price_ok: true }
  },
  {
    product_id: "ADI-00009",
    name: "Own the Run Jacket",
    category: "Apparel",
    subcategory: "Jackets",
    color: "Black",
    color_details: "Black",
    selling_price: 75.00,
    original_price: 85.00,
    brand: "Adidas",
    rating: 4.6,
    image_url: "https://images.unsplash.com/photo-1556905055-8f358a7a47b2?auto=format&fit=crop&w=600&q=80",
    similarity_score: 0.785,
    rerank_score: 0.78,
    initial_rank: 1,
    final_rank: 9,
    constraint_status: { all_satisfied: false, category_ok: false, color_ok: true, price_ok: true, status_text: "Price marginal" }
  },
  {
    product_id: "ADI-00010",
    name: "Own the Run Shorts",
    category: "Apparel",
    subcategory: "Shorts",
    color: "Black",
    color_details: "Black",
    selling_price: 40.00,
    original_price: 45.00,
    brand: "Adidas",
    rating: 4.5,
    image_url: "https://images.unsplash.com/photo-1591195853828-11db59a44f6b?auto=format&fit=crop&w=600&q=80",
    similarity_score: 0.772,
    rerank_score: 0.78,
    initial_rank: 3,
    final_rank: 10,
    constraint_status: { all_satisfied: true, category_ok: false, color_ok: true, price_ok: true }
  }
];

let currentUploadedImageBase64 = null;
let currentResults = [...INITIAL_DEMO_RESULTS];

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initThemeToggle();
  initPresetChips();
  initImageUpload();
  initSearch();
  initSortingAndViews();
  renderProductCards(currentResults);
  fetchExperimentsData();
});

// 1. Tab Navigation
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const targetId = tab.getAttribute("data-tab");
      document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.remove("active");
      });

      const pane = document.getElementById(targetId.replace("-tab", "-pane"));
      if (pane) pane.classList.add("active");
    });
  });
}

// 2. Theme Switcher
function initThemeToggle() {
  const toggleBtn = document.getElementById("theme-toggle");
  const moonIcon = toggleBtn.querySelector(".moon-icon");
  const sunIcon = toggleBtn.querySelector(".sun-icon");

  const savedTheme = localStorage.getItem("cs_theme") || "light";
  if (savedTheme === "dark") {
    document.body.classList.remove("light-mode");
    document.body.classList.add("dark-mode");
    moonIcon.style.display = "none";
    sunIcon.style.display = "block";
  }

  toggleBtn.addEventListener("click", () => {
    if (document.body.classList.contains("dark-mode")) {
      document.body.classList.remove("dark-mode");
      document.body.classList.add("light-mode");
      moonIcon.style.display = "block";
      sunIcon.style.display = "none";
      localStorage.setItem("cs_theme", "light");
    } else {
      document.body.classList.remove("light-mode");
      document.body.classList.add("dark-mode");
      moonIcon.style.display = "none";
      sunIcon.style.display = "block";
      localStorage.setItem("cs_theme", "dark");
    }
  });
}

// 3. Preset Chips
function initPresetChips() {
  const chips = document.querySelectorAll(".preset-chip");
  const queryInput = document.getElementById("query-input");

  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-query");
      queryInput.value = q;
      triggerSearch();
    });
  });

  const clearBtn = document.getElementById("btn-clear-query");
  clearBtn.addEventListener("click", () => {
    queryInput.value = "";
    queryInput.focus();
  });
}

// 4. Image Upload & Dropzone
function initImageUpload() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const uploadPrompt = document.getElementById("upload-prompt");
  const previewBox = document.getElementById("image-preview-box");
  const previewImg = document.getElementById("ref-image-preview");
  const removeBtn = document.getElementById("btn-remove-image");
  const fileNameSpan = document.getElementById("ref-img-name");
  const fileSizeSpan = document.getElementById("ref-img-size");

  dropZone.addEventListener("click", (e) => {
    if (e.target !== removeBtn) fileInput.click();
  });

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) handleImageFile(file);
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-hover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-hover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-hover");
    if (e.dataTransfer.files.length > 0) {
      handleImageFile(e.dataTransfer.files[0]);
    }
  });

  removeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    currentUploadedImageBase64 = null;
    previewImg.src = "";
    previewBox.style.display = "none";
    uploadPrompt.style.display = "flex";
    fileNameSpan.innerText = "No image selected";
    fileSizeSpan.innerText = "0 KB";
  });

  function handleImageFile(file) {
    fileNameSpan.innerText = file.name;
    fileSizeSpan.innerText = `${Math.round(file.size / 1024)} KB`;
    
    const reader = new FileReader();
    reader.onload = (evt) => {
      currentUploadedImageBase64 = evt.target.result;
      previewImg.src = currentUploadedImageBase64;
      previewBox.style.display = "flex";
      uploadPrompt.style.display = "none";
      triggerSearch();
    };
    reader.readAsDataURL(file);
  }
}

// 5. Search Execution
function initSearch() {
  const searchBtn = document.getElementById("btn-search");
  const queryInput = document.getElementById("query-input");

  searchBtn.addEventListener("click", triggerSearch);
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") triggerSearch();
  });
}

async function triggerSearch() {
  const query = document.getElementById("query-input").value.trim();
  if (!query && !currentUploadedImageBase64) return;

  const btn = document.getElementById("btn-search");
  btn.innerHTML = `<span class="spinner"></span> Searching...`;
  btn.disabled = true;

  try {
    let payload = { query: query, top_k: 10 };
    let endpoint = "/api/search/text";

    if (currentUploadedImageBase64) {
      payload.image_base64 = currentUploadedImageBase64;
      endpoint = "/api/search/multimodal";
    }

    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      const data = await res.json();
      currentResults = data.results || [];
      renderProductCards(currentResults);
      updateConstraintsUI(data.extracted_constraints);
      updatePipelineTimingsUI(data.timings);
      updateEvaluationUI(data.evaluation, data.baseline_comparison);
    } else {
      renderClientFallback(query);
    }
  } catch (err) {
    renderClientFallback(query);
  } finally {
    btn.innerHTML = `<svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" fill="none" stroke-width="2.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg> Search`;
    btn.disabled = false;
  }
}

function renderClientFallback(query) {
  const isBlack = query.toLowerCase().includes("black");
  const isWhite = query.toLowerCase().includes("white");
  const isBlue = query.toLowerCase().includes("blue");
  const color = isBlack ? "Black" : isWhite ? "White" : isBlue ? "Blue" : "Black";
  
  let maxPrice = 120.0;
  const priceMatch = query.match(/(?:under|below|\$)\s*(\d+)/i);
  if (priceMatch) maxPrice = parseFloat(priceMatch[1]);

  const cat = query.toLowerCase().includes("jacket") ? "Apparel" : "Footwear";

  updateConstraintsUI({
    category: cat,
    color: color,
    max_price: maxPrice,
    brand: "Adidas"
  });

  updatePipelineTimingsUI({
    clip_encoding_ms: 18.4,
    pinecone_retrieval_ms: 8.2,
    constraint_parsing_ms: 0.4,
    lambdamart_reranking_ms: 2.1,
    total_ms: 29.1
  });

  renderProductCards(currentResults);
}

// 6. Product Cards Rendering
function renderProductCards(results) {
  const grid = document.getElementById("product-grid");
  const countPill = document.getElementById("results-count");
  
  grid.innerHTML = "";
  countPill.innerText = `${results.length} results`;

  results.forEach((prod, idx) => {
    const rank = prod.final_rank || (idx + 1);
    const score = (prod.rerank_score !== undefined ? prod.rerank_score : 0.95).toFixed(2);
    const status = prod.constraint_status || { all_satisfied: true };

    let statusHtml = "";
    if (status.all_satisfied) {
      statusHtml = `<span class="prod-status-tag status-tag-pass">${score} All constraints ✓</span>`;
    } else if (status.status_text === "Price marginal" || (status.price_ok === false && status.category_ok)) {
      statusHtml = `<span class="prod-status-tag status-tag-marginal">${score} Price marginal</span>`;
    } else {
      statusHtml = `<span class="prod-status-tag status-tag-fail">${score} Violated constraint</span>`;
    }

    const isApparel = (prod.category || "").toLowerCase() === "apparel";
    const fallbackIcon = isApparel ?
      `<svg viewBox="0 0 24 24" width="44" height="44" stroke="#0d9488" fill="none" stroke-width="1.5"><path d="M20.38 3.46L16 2a4 4 0 0 1-8 0L3.62 3.46a2 2 0 0 0-1.34 2.23l.58 3.47a1 1 0 0 0 .99.84H6v10c0 1.1.9 2 2 2h8a2 2 0 0 0 2-2V10h2.15a1 1 0 0 0 .99-.84l.58-3.47a2 2 0 0 0-1.34-2.23z"/></svg>` :
      `<svg viewBox="0 0 24 24" width="44" height="44" stroke="#0d9488" fill="none" stroke-width="1.5"><path d="M4 16v-2a4 4 0 0 1 4-4h8a4 4 0 0 1 4 4v2"/><path d="M2 16h20v4a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-4z"/><circle cx="8" cy="14" r="1"/></svg>`;

    const card = document.createElement("div");
    card.className = "product-card";
    card.innerHTML = `
      <div class="card-rank-badge">${rank}</div>
      <div class="product-img-wrapper">
        <img src="${prod.image_url}" alt="${prod.name}" loading="lazy" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
        <div class="product-fallback-visual" style="display:none; width:100%; height:100%; align-items:center; justify-content:center; background:linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); flex-direction:column; gap:0.25rem;">
          ${fallbackIcon}
          <span style="font-size:0.65rem; color:#64748b; font-weight:600;">${prod.color || 'Adidas'} • ${prod.subcategory || prod.category}</span>
        </div>
      </div>
      <div class="product-card-body">
        <h4 class="prod-title">${prod.name}</h4>
        <div class="prod-price">$${Number(prod.selling_price).toFixed(2)}</div>
        <div class="prod-variant">${prod.color_details || prod.color}</div>
        ${statusHtml}
      </div>
    `;
    grid.appendChild(card);
  });
}

function updateConstraintsUI(c) {
  if (!c) return;
  if (c.category) document.getElementById("c-cat").innerText = c.category;
  if (c.color) document.getElementById("c-col").innerText = c.color;
  if (c.max_price) document.getElementById("c-price").innerText = `<$${c.max_price}`;
  if (c.brand) document.getElementById("c-brand").innerText = c.brand;
}

function updatePipelineTimingsUI(t) {
  if (!t) return;
  if (t.clip_encoding_ms) document.getElementById("time-clip").innerText = `${t.clip_encoding_ms}ms`;
  if (t.pinecone_retrieval_ms) document.getElementById("time-pinecone").innerText = `${t.pinecone_retrieval_ms}ms`;
  if (t.constraint_parsing_ms) document.getElementById("time-parsing").innerText = `${t.constraint_parsing_ms}ms`;
  if (t.lambdamart_reranking_ms) document.getElementById("time-lambdamart").innerText = `${t.lambdamart_reranking_ms}ms`;
  if (t.total_ms) document.getElementById("total-exec-time").innerText = `${t.total_ms}ms`;
}

function updateEvaluationUI(evalData, compData) {
  if (!evalData) return;
  document.getElementById("m-ndcg").innerText = evalData.ndcg_at_10.toFixed(3);
  document.getElementById("m-recall").innerText = evalData.recall_at_10.toFixed(3);
  document.getElementById("m-cs").innerText = evalData.constraint_satisfaction_rate.toFixed(3);
  document.getElementById("m-lat").innerText = `${evalData.p95_latency_s.toFixed(2)} s`;
}

// 7. Sorting & Grid/List View
function initSortingAndViews() {
  const sortSelect = document.getElementById("sort-select");
  sortSelect.addEventListener("change", () => {
    const val = sortSelect.value;
    if (val === "price_low") {
      currentResults.sort((a, b) => a.selling_price - b.selling_price);
    } else if (val === "rating") {
      currentResults.sort((a, b) => (b.rating || 0) - (a.rating || 0));
    } else if (val === "similarity") {
      currentResults.sort((a, b) => (b.similarity_score || 0) - (a.similarity_score || 0));
    } else {
      currentResults.sort((a, b) => (b.rerank_score || 0) - (a.rerank_score || 0));
    }
    renderProductCards(currentResults);
  });

  const btnGrid = document.getElementById("btn-view-grid");
  const btnList = document.getElementById("btn-view-list");
  const grid = document.getElementById("product-grid");

  btnGrid.addEventListener("click", () => {
    btnGrid.classList.add("active");
    btnList.classList.remove("active");
    grid.classList.remove("list-view");
  });

  btnList.addEventListener("click", () => {
    btnList.classList.add("active");
    btnGrid.classList.remove("active");
    grid.classList.add("list-view");
  });
}

// 8. Fetch Experiments Data
async function fetchExperimentsData() {
  try {
    const res = await fetch("/api/experiments");
    if (res.ok) {
      const data = await res.json();
      if (data.percentage_improvements) {
        document.getElementById("exp-ndcg-gain").innerText = `+${data.percentage_improvements["ndcg@10"].toFixed(1)}%`;
        document.getElementById("exp-cs-gain").innerText = `+${data.percentage_improvements["constraint_satisfaction"].toFixed(1)}%`;
        document.getElementById("exp-mc-gain").innerText = `+${data.percentage_improvements["multi_constraint_success"].toFixed(1)}%`;
      }
      if (data.proposed_system && data.proposed_system.latency_ms) {
        document.getElementById("exp-p95-lat").innerText = `${data.proposed_system.latency_ms.p95} ms`;
      }
    }
  } catch (e) {
  }
}
