
(() => {
  const slides = [...document.querySelectorAll('.slide')];
  if (!slides.length) return;
  const previous = document.querySelector('#previous');
  const next = document.querySelector('#next');
  const toggle = document.querySelector('#toggle');
  const progress = document.querySelector('#progress');
  let current = 0;
  function render() {
    const all = location.hash === '#all';
    const requested = Number(location.hash.slice(1));
    if (!all && Number.isInteger(requested) && requested >= 1 && requested <= slides.length) current = requested - 1;
    document.body.classList.toggle('all', all);
    if (!all) window.scrollTo(0, 0);
    slides.forEach((slide, i) => slide.hidden = !all && i !== current);
    previous.disabled = all || current === 0;
    next.disabled = all || current === slides.length - 1;
    toggle.textContent = all ? 'Slide view' : 'Show all';
    progress.textContent = all ? `${slides.length} slides` : `${current + 1} / ${slides.length}`;
  }
  function move(delta) {
    if (location.hash === '#all') return;
    current = Math.max(0, Math.min(slides.length - 1, current + delta));
    location.hash = String(current + 1);
    render();
  }
  previous.addEventListener('click', () => move(-1));
  next.addEventListener('click', () => move(1));
  toggle.addEventListener('click', () => location.hash = location.hash === '#all' ? String(current + 1) : 'all');
  document.addEventListener('keydown', (event) => {
    if (event.target.closest('button,a,input,select,textarea')) return;
    if (['ArrowRight','PageDown',' '].includes(event.key)) { event.preventDefault(); move(1); }
    if (['ArrowLeft','PageUp'].includes(event.key)) { event.preventDefault(); move(-1); }
    if (event.key === 'Home') location.hash = '1';
    if (event.key === 'End') location.hash = String(slides.length);
  });
  window.addEventListener('hashchange', render);
  document.body.classList.add('deck');
  render();
})();
