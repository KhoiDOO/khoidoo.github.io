// Smooth scrolling for internal links
document.querySelectorAll('a[href^="#"]').forEach(a=>{
  a.addEventListener('click',e=>{
    const href=a.getAttribute('href');
    if(href.length>1){
      e.preventDefault();
      const el = document.querySelector(href);
      if(el) el.scrollIntoView({behavior:'smooth',block:'start'});
    }
  })
});

// Helper: you can change the photo path or other small behaviors here.

// Collapse news list to first N items and toggle with the "More..." link
// Collapse news list to first N items and toggle with the "More..." link
function initNewsToggle(){
  const N = 5; // number of news items to show initially
  const newsSection = document.getElementById('news');
  if(!newsSection) return;

  const allParagraphs = Array.from(newsSection.querySelectorAll('p'));
  // find the "More..." paragraph (contains link to news.html)
  const morePara = allParagraphs.find(p => p.querySelector && p.querySelector('a[href="news.html"]'));
  const newsItems = allParagraphs.filter(p => p !== morePara);

  if(newsItems.length <= N){
    // nothing to collapse, hide the More link if present
    if(morePara) morePara.style.display = 'none';
    return;
  }

  // hide items beyond the first N
  const hidden = newsItems.slice(N);
  hidden.forEach(p => p.style.display = 'none');

  if(morePara){
    const moreLink = morePara.querySelector('a[href="news.html"]');
    if(moreLink){
      moreLink.addEventListener('click', e => {
        e.preventDefault();
        // reveal hidden items: clear display so CSS applies and force left alignment
        hidden.forEach(p => {
          p.style.display = '';
          p.style.justifyContent = 'flex-start';
          p.style.textAlign = 'left';
        });
        // remove the More paragraph
        morePara.style.display = 'none';

        // hide other sections (About and Publications)
        const about = document.getElementById('about');
        const pubs = document.getElementById('publications');
        if(about) about.style.display = 'none';
        if(pubs) pubs.style.display = 'none';

        // update ARIA
        moreLink.setAttribute('aria-expanded','true');

        // scroll to the first newly revealed item
        hidden[0].scrollIntoView({behavior:'smooth',block:'start'});
      });
      // ensure the link does not navigate if JS runs after click
      moreLink.setAttribute('role','button');
      moreLink.setAttribute('aria-expanded','false');
    }
  }
}

// Run immediately if DOM already loaded, otherwise wait for DOMContentLoaded
if(document.readyState === 'loading'){
  document.addEventListener('DOMContentLoaded', initNewsToggle);
} else {
  initNewsToggle();
}
