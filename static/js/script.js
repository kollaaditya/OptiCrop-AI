/* ============================================================
   OptiCrop AI – Main JavaScript
   ============================================================ */

// ── Dark Mode ──────────────────────────────────────────────
const html      = document.documentElement;
const themeBtn  = document.getElementById('themeToggle');
const themeIcon = document.getElementById('themeIcon');

function applyTheme(theme) {
  html.setAttribute('data-theme', theme);
  localStorage.setItem('opticrop-theme', theme);
  if (themeIcon) {
    themeIcon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
  }
}

// Load saved theme
applyTheme(localStorage.getItem('opticrop-theme') || 'light');

if (themeBtn) {
  themeBtn.addEventListener('click', () => {
    applyTheme(html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  });
}

// ── Counter Animation ──────────────────────────────────────
function animateCounter(el) {
  const target   = parseInt(el.dataset.target, 10);
  const duration = 1800;
  const step     = target / (duration / 16);
  let current    = 0;

  const timer = setInterval(() => {
    current += step;
    if (current >= target) {
      el.textContent = target.toLocaleString();
      clearInterval(timer);
    } else {
      el.textContent = Math.floor(current).toLocaleString();
    }
  }, 16);
}

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      animateCounter(entry.target);
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.5 });

document.querySelectorAll('.counter').forEach(el => observer.observe(el));

// ── Auto-dismiss Alerts ────────────────────────────────────
document.querySelectorAll('.alert').forEach(alert => {
  setTimeout(() => {
    const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
    if (bsAlert) bsAlert.close();
  }, 5000);
});

// ── Navbar Scroll Effect ───────────────────────────────────
window.addEventListener('scroll', () => {
  const nav = document.querySelector('.navbar-custom');
  if (nav) {
    nav.style.boxShadow = window.scrollY > 20
      ? '0 4px 20px rgba(0,0,0,0.1)'
      : 'none';
  }
});

// ── Smooth Scroll ──────────────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const target = document.querySelector(a.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// ── Form Validation Feedback ───────────────────────────────
document.querySelectorAll('input[required], textarea[required]').forEach(input => {
  input.addEventListener('blur', () => {
    if (!input.value.trim()) {
      input.classList.add('is-invalid');
    } else {
      input.classList.remove('is-invalid');
      input.classList.add('is-valid');
    }
  });
  input.addEventListener('input', () => {
    if (input.value.trim()) {
      input.classList.remove('is-invalid');
    }
  });
});

// ── Tooltip Init ───────────────────────────────────────────
document.querySelectorAll('[title]').forEach(el => {
  new bootstrap.Tooltip(el, { trigger: 'hover' });
});
