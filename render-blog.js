const posts = [
  'control-theory.md',
  // Add more blog files here
];

const blogList = document.getElementById('blog-list');

posts.forEach(post => {
  fetch(`blog-posts/${post}`)
    .then(res => res.text())
    .then(text => {
      // Extract title
      const titleMatch = text.match(/^#\s+(.+)/m);
      const title = titleMatch ? titleMatch[1].trim() : 'Untitled';

      // Extract image URL
      const imageMatch = text.match(/<!--\s*image:\s*(.*?)\s*-->/);
      const imageURL = imageMatch ? imageMatch[1].trim() : '';

      // Extract summary (skip lines starting with # or <!--)
      const lines = text.split('\n');
      const contentLines = lines.filter(line =>
        !line.trim().startsWith('#') &&
        !line.trim().startsWith('<!--') &&
        line.trim().length > 0
      );
      const summaryLines = contentLines.slice(0, 7).join(' ');
      const summaryHTML = marked.parseInline(summaryLines);

      // Create slug
      const postSlug = post.replace('.md', '');

      // Build image block
      const imageHTML = imageURL
        ? `
          <div style="height: 100%;">
            <img src="${imageURL}" alt="Thumbnail"
              style="height: 100%; width: 100%; object-fit: cover; border-radius: 4px;" />
          </div>
        `
        : '';

      // Build blog card
      const blogHTML = `
        <div class="blog-card" style="display: flex; background: #eee; padding: 20px; margin-bottom: 20px; border-radius: 6px; align-items: stretch;">
          <div style="flex: 0 0 160px; margin-right: 20px;">
            ${imageHTML}
          </div>
          <div style="flex: 1; display: flex; flex-direction: column;">
            <h3 style="margin-bottom: 10px;"><a href="blog-template.html?post=${postSlug}">${title}</a></h3>
            <p style="font-size: 15px; line-height: 1.5; flex: 1;">
              ${summaryHTML}... <a href="blog-template.html?post=${postSlug}">See more</a>
            </p>
          </div>
        </div>
      `;

      blogList.innerHTML += blogHTML;
    })
    .catch(error => {
      console.error(`Error loading post ${post}:`, error);
    });
});
