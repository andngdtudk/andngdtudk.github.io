const posts = [
  'control-theory.md',
  // Add more blog files here
];

const blogList = document.getElementById('blog-list');

posts.forEach(post => {
  fetch(`blog-posts/${post}`)
    .then(res => res.text())
    .then(text => {
      const titleMatch = text.match(/^#\s+(.+)/m);
      const imageMatch = text.match(/<!--\s*image:\s*(.*?)\s*-->/);
      const title = titleMatch ? titleMatch[1].trim() : 'Untitled';

      // Extract the first few lines of content that aren't titles or comments
      const lines = text.split('\n');
      const contentLines = lines.filter(line =>
        !line.trim().startsWith('#') &&
        !line.trim().startsWith('<!--') &&
        line.trim() !== ''
      );
      const summaryLines = contentLines.slice(0, 7).join(' ');
      const summaryHTML = marked.parseInline(summaryLines);

      const postSlug = post.replace('.md', '');
      const imageURL = imageMatch ? imageMatch[1].trim() : '';
      const imageHTML = imageURL
        ? `
          <div style="height: 100%; display: flex;">
            <img src="${imageURL}" alt="Thumbnail"
                 style="width: 160px; height: 100%; object-fit: cover; border-radius: 4px;" />
          </div>
        `
        : '';

      const blogHTML = `
        <div class="blog-card" style="display: flex; background: #eee; padding: 20px; margin-bottom: 20px; border-radius: 6px;">
          <div style="flex: 0 0 160px; margin-right: 20px; display: flex; align-items: stretch;">
            ${imageHTML}
          </div>
          <div style="flex: 1; display: flex; flex-direction: column; justify-content: space-between;">
            <h3 style="margin-bottom: 10px;">
              <a href="blog-template.html?post=${postSlug}">${title}</a>
            </h3>
            <p class="blog-summary">${summaryHTML}... <a href="blog-template.html?post=${postSlug}">See more</a></p>
          </div>
        </div>
      `;

      blogList.innerHTML += blogHTML;
    });
});
