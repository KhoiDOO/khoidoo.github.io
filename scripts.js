/**
 * Khoidoo Academic Homepage - Advanced Interactivity & Filtering Suite
 * Features:
 * - Smooth anchor scrolling & ScrollSpy navigation
 * - News items toggle / progressive disclosure
 * - Expandable BibTeX drawers with instant clipboard copy
 * - Scalable Publication Filtering (Topic, Year & Keyword Search)
 * - Scalable Project Filtering (Topic Tabs)
 * - Floating Back-to-Top button with scroll observer
 */

// Smooth scrolling for all internal anchors
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const href = a.getAttribute('href');
      if (href && href.length > 1) {
        const targetEl = document.querySelector(href);
        if (targetEl) {
          e.preventDefault();
          targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }
    });
  });
}

// ScrollSpy to highlight active navigation link
function initScrollSpy() {
  const navPills = document.querySelectorAll('.quick-nav .nav-pill');
  const sections = document.querySelectorAll('main.content section[id]');
  if (!navPills.length || !sections.length) return;

  const observerOptions = {
    root: null,
    rootMargin: '-20% 0px -70% 0px',
    threshold: 0
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.getAttribute('id');
        navPills.forEach(pill => {
          if (pill.getAttribute('href') === `#${id}`) {
            pill.classList.add('active');
          } else {
            pill.classList.remove('active');
          }
        });
      }
    });
  }, observerOptions);

  sections.forEach(section => observer.observe(section));
}

// News toggle controller (collapses items beyond N)
function initNewsToggle() {
  const N = 4; // number of news items to show initially
  const newsList = document.querySelector('.news-list');
  const toggleWrap = document.querySelector('.news-toggle-wrap');
  const toggleBtn = document.querySelector('.more-news-btn');
  
  if (!newsList) return;

  const newsItems = Array.from(newsList.querySelectorAll('.news-item'));
  if (newsItems.length <= N) {
    if (toggleWrap) toggleWrap.style.display = 'none';
    return;
  }

  // Hide trailing items
  const hiddenItems = newsItems.slice(N);
  hiddenItems.forEach(item => {
    item.style.display = 'none';
  });

  if (toggleBtn) {
    let expanded = false;
    toggleBtn.addEventListener('click', e => {
      e.preventDefault();
      expanded = !expanded;
      
      hiddenItems.forEach(item => {
        item.style.display = expanded ? 'flex' : 'none';
      });

      toggleBtn.innerHTML = expanded ? 'Show Less &uarr;' : 'More News &rarr;';
      toggleBtn.setAttribute('aria-expanded', String(expanded));

      if (expanded && hiddenItems[0]) {
        hiddenItems[0].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    });
  }
}

// Publication Filtering & Live Search Engine
function initPubFilters() {
  const pubCards = document.querySelectorAll('.pub-card');
  const topicFilterBtns = document.querySelectorAll('#pub-topic-filters .filter-btn');
  const yearFilterBtns = document.querySelectorAll('#pub-year-filters .filter-btn');
  const searchInput = document.getElementById('pub-search-input');
  const noResultsMsg = document.getElementById('pub-no-results');

  if (!pubCards.length) return;

  let currentTopic = 'all';
  let currentYear = 'all';
  let currentSearch = '';

  function applyFilters() {
    let visibleCount = 0;
    const query = currentSearch.toLowerCase().trim();

    pubCards.forEach(card => {
      const topic = card.getAttribute('data-topic') || '';
      const year = card.getAttribute('data-year') || '';
      const content = card.textContent.toLowerCase();

      const matchesTopic = (currentTopic === 'all') || topic.includes(currentTopic);
      const matchesYear = (currentYear === 'all') || (year === currentYear);
      const matchesSearch = !query || content.includes(query);

      if (matchesTopic && matchesYear && matchesSearch) {
        card.classList.remove('is-hidden');
        visibleCount++;
      } else {
        card.classList.add('is-hidden');
      }
    });

    if (noResultsMsg) {
      noResultsMsg.style.display = visibleCount === 0 ? 'block' : 'none';
    }
  }

  // Topic filter events
  topicFilterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      topicFilterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentTopic = btn.getAttribute('data-filter') || 'all';
      applyFilters();
    });
  });

  // Year filter events
  yearFilterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      yearFilterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentYear = btn.getAttribute('data-filter') || 'all';
      applyFilters();
    });
  });

  // Live search events
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      currentSearch = e.target.value;
      applyFilters();
    });
  }
}

// Project Filtering Engine
function initProjectFilters() {
  const projectCards = document.querySelectorAll('.project-card');
  const filterBtns = document.querySelectorAll('#project-filters .filter-btn');

  if (!projectCards.length || !filterBtns.length) return;

  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const targetTopic = btn.getAttribute('data-filter') || 'all';

      projectCards.forEach(card => {
        const topic = card.getAttribute('data-topic') || '';
        if (targetTopic === 'all' || topic.includes(targetTopic)) {
          card.classList.remove('is-hidden');
        } else {
          card.classList.add('is-hidden');
        }
      });
    });
  });
}

// BibTeX drawer toggle and clipboard copy handler
function initBibtex() {
  // Toggle drawers
  document.querySelectorAll('.bibtex-trigger').forEach(btn => {
    btn.addEventListener('click', e => {
      e.preventDefault();
      const targetId = btn.getAttribute('data-target');
      if (!targetId) return;
      const drawer = document.getElementById(targetId);
      if (drawer) {
        const isCurrentlyActive = drawer.classList.contains('active');
        drawer.classList.toggle('active');
        btn.setAttribute('aria-expanded', String(!isCurrentlyActive));
      }
    });
  });

  // Copy to clipboard
  document.querySelectorAll('.copy-bibtex-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      e.preventDefault();
      const rawBibtex = btn.getAttribute('data-bibtex');
      if (!rawBibtex) {
        const parent = btn.closest('.bibtex-drawer');
        const codeEl = parent ? parent.querySelector('.bibtex-code') : null;
        if (codeEl) {
          copyTextToClipboard(codeEl.textContent.trim(), btn);
        }
        return;
      }
      copyTextToClipboard(rawBibtex.trim(), btn);
    });
  });
}

function copyTextToClipboard(text, btnElement) {
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(() => {
      showCopySuccess(btnElement);
    }).catch(() => {
      fallbackCopy(text, btnElement);
    });
  } else {
    fallbackCopy(text, btnElement);
  }
}

function fallbackCopy(text, btnElement) {
  const textArea = document.createElement('textarea');
  textArea.value = text;
  textArea.style.position = 'fixed';
  textArea.style.left = '-999999px';
  textArea.style.top = '-999999px';
  document.body.appendChild(textArea);
  textArea.focus();
  textArea.select();
  try {
    document.execCommand('copy');
    showCopySuccess(btnElement);
  } catch (err) {
    console.error('Copy failed:', err);
  }
  document.body.removeChild(textArea);
}

function showCopySuccess(btn) {
  if (!btn) return;
  const originalText = btn.innerHTML;
  btn.innerHTML = 'Copied! &#10003;';
  btn.classList.add('copied');
  setTimeout(() => {
    btn.innerHTML = originalText;
    btn.classList.remove('copied');
  }, 2000);
}

// Back-to-top floating button controller
function initBackToTop() {
  const backToTopBtn = document.getElementById('backToTop');
  if (!backToTopBtn) return;

  window.addEventListener('scroll', () => {
    if (window.scrollY > 300) {
      backToTopBtn.classList.add('show');
    } else {
      backToTopBtn.classList.remove('show');
    }
  }, { passive: true });

  backToTopBtn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}

// Initialize all features on DOM ready
function initAll() {
  initSmoothScroll();
  initScrollSpy();
  initNewsToggle();
  initPubFilters();
  initProjectFilters();
  initBibtex();
  initBackToTop();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initAll);
} else {
  initAll();
}
