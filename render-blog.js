const posts = [
  'control-theory.md',
  // Add more posts here
];

const blogList = document.getElementById('blog-list');

posts.forEach(post => {
  fetch(`blog-posts/${post}`)
    .then(res => res.text())
    .then(text => {
      const titleMatch = text.match(/^#\s+(.+)/m);
      const imageMatch = text.match(/<!--\s*image:\s*(.*?)\s*-->/);
      const title = titleMatch ? titleMatch[1].trim() : 'Untitled';

      const lines = text.split('\n');
      const contentLines = lines.filter(line => !line.trim().startsWith('#') && !line.trim().startsWith('<!--'));
      const summaryLines = contentLines.slice(0, 7).join(' ');
      const summaryHTML = marked.parseInline(summaryLines);

      const postSlug = post.replace('.md', '');
      const imageURL = imageMatch ? imageMatch[1].trim() : '';
      const imageHTML = imageURL
        ? `<img src="${imageURL}" alt="Thumbnail" style="width: 160px; height: 160px; object-fit: cover; border-radius: 4px;" />`
        : '';

      const blogHTML = `
        <div class="blog-card" style="display: flex; align-items: flex-start; background: #eee; padding: 20px; margin-bottom: 20px; border-radius: 6px;">
          <div style="flex: 0 0 160px; margin-right: 20px;">
            ${imageHTML}
          </div>
          <div style="flex: 1;">
            <h3 style="margin-bottom: 10px;"><a href="blog-template.html?post=${postSlug}">${title}</a></h3>
            <p style="font-size: 15px; line-height: 1.5;">${summaryHTML}... <a href="blog-template.html?post=${postSlug}">See more</a></p>
          </div>
        </div>
      `;

      blogList.innerHTML += blogHTML;
    });
});
