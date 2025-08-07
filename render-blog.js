const posts = [
  'control-theory.md',
  // Add more markdown files here, e.g., 'mpc-intro.md'
];

const blogList = document.getElementById('blog-list');

posts.forEach(post => {
  fetch(`blog-posts/${post}`)
    .then(res => res.text())
    .then(text => {
      const titleMatch = text.match(/^# (.+)/);
      const imageMatch = text.match(/<!--\s*image:\s*(.*?)\s*-->/);
      const title = titleMatch ? titleMatch[1] : 'Untitled';
      const summary = text.split('\n').slice(1, 5).join(' ');
      const postSlug = post.replace('.md', '');
      const imageHTML = imageMatch
        ? `<img src="images/${imageMatch[1]}" alt="Thumbnail" style="width: 120px; border-radius: 4px;" />`
        : '';

      const blogHTML = `
        <div class="blog-card" style="display: flex; background: #eee; padding: 20px; margin-bottom: 20px; border-radius: 6px;">
          <div style="flex: 0 0 120px; margin-right: 20px;">
            ${imageHTML}
          </div>
          <div style="flex: 1;">
            <h3 style="margin-bottom: 10px;"><a href="blog-template.html?post=${postSlug}">${title}</a></h3>
            <p style="font-size: 15px; line-height: 1.5;">${summary}... <a href="blog-template.html?post=${postSlug}">See more</a></p>
          </div>
        </div>
      `;

      blogList.innerHTML += blogHTML;
    });
});
