document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('search-form');
  const input = document.getElementById('query-input');
  const likesSlider = document.getElementById('min-likes');
  const likesValue = document.getElementById('likes-value');
  
  const loadingEl = document.getElementById('loading');
  const errorEl = document.getElementById('error');
  const errorMessage = document.getElementById('error-message');
  
  const resultsEl = document.getElementById('results');
  const answerContent = document.getElementById('answer-content');
  const latencyEl = document.getElementById('latency');
  const sourcesGrid = document.getElementById('sources-grid');

  // Update slider badge
  likesSlider.addEventListener('input', (e) => {
    likesValue.textContent = e.target.value;
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    // Reset UI state
    loadingEl.classList.remove('hidden');
    resultsEl.classList.add('hidden');
    errorEl.classList.add('hidden');
    
    // Clear previous results
    answerContent.innerHTML = '';
    sourcesGrid.innerHTML = '';
    
    try {
      // Perform request
      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: query,
          min_likes: parseInt(likesSlider.value, 10),
          top_k: 5
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Render output
      renderResults(data);
      
    } catch (err) {
      console.error(err);
      errorMessage.textContent = `Error: ${err.message}. Make sure the LUMINA backend is running.`;
      errorEl.classList.remove('hidden');
    } finally {
      loadingEl.classList.add('hidden');
    }
  });

  function renderResults(data) {
    // 1. Parse Markdown answer
    answerContent.innerHTML = marked.parse(data.answer || "No answer generated.");
    
    // 2. Render Latency
    latencyEl.textContent = `⏱️ ${(data.latency_ms / 1000).toFixed(2)}s`;
    
    // 3. Render Sources
    const sources = data.retrieved_posts || [];
    if (sources.length === 0) {
      sourcesGrid.innerHTML = '<p style="color: #888">No sources found.</p>';
    } else {
      sourcesGrid.innerHTML = sources.map((source, index) => `
        <div class="source-card" style="cursor: pointer;" data-index="${index}">
          <div class="source-header">
            <div class="author">
              <div class="author-avatar"></div>
              @${escapeHtml(source.username)}
            </div>
            <div class="likes">
              ❤️ ${source.likes.toLocaleString()}
            </div>
          </div>
          <div class="image-container">
            <img src="http://localhost:8000/image/${escapeHtml(source.post_id)}" alt="Post image" loading="lazy" />
          </div>
          <div class="source-text">
            ${escapeHtml(source.text_chunk)}
          </div>
          <div style="margin-top: 10px; font-size: 0.85rem; color: var(--ig-primary); font-weight: 600;">
            Click to view full post ↗
          </div>
        </div>
      `).join('');

      // Add click listeners to open modal
      document.querySelectorAll('.source-card').forEach(card => {
        card.addEventListener('click', () => {
          const idx = parseInt(card.getAttribute('data-index'));
          openModal(sources[idx]);
        });
      });
    }
    
    // Show results
    resultsEl.classList.remove('hidden');
  }

  // Modal Logic
  let modal = document.getElementById('post-modal');
  
  // If index.html is cached and modal doesn't exist, inject it dynamically
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'post-modal';
    modal.className = 'modal hidden';
    modal.innerHTML = `
      <div class="modal-content glass">
        <span class="close-btn">&times;</span>
        <div class="modal-header">
          <div class="author">
            <div class="author-avatar"></div>
            <span id="modal-username"></span>
          </div>
          <div class="likes" id="modal-likes"></div>
        </div>
        <div class="modal-image-container">
          <img id="modal-image" src="" alt="Post image" />
        </div>
        <div class="modal-text" id="modal-text"></div>
      </div>
    `;
    document.body.appendChild(modal);
  }

  const closeBtn = modal.querySelector('.close-btn');
  const modalUsername = document.getElementById('modal-username');
  const modalLikes = document.getElementById('modal-likes');
  const modalImage = document.getElementById('modal-image');
  const modalText = document.getElementById('modal-text');

  function openModal(source) {
    modalUsername.textContent = '@' + source.username;
    modalLikes.textContent = '❤️ ' + source.likes.toLocaleString();
    modalImage.src = `http://localhost:8000/image/${source.post_id}`;
    modalText.textContent = source.text_chunk;
    modal.classList.remove('hidden');
  }

  function closeModal() {
    modal.classList.add('hidden');
  }

  closeBtn.addEventListener('click', closeModal);
  modal.addEventListener('click', (e) => {
    if (e.target === modal) closeModal();
  });

  // Simple HTML escaper
  function escapeHtml(unsafe) {
    return (unsafe || '')
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
