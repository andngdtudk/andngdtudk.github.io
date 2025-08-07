const posts = [
  'control-theory.md',
  // Add more blog files here
];

const blogList = document.getElementById('blog-list');

posts.forEach(post => {
  fetch(`/blog-posts/${post}`)
    .then(res => res.text())
    .then(text => {
      const titleMatch = text.match(/^#\s+(.+)/m);
      const imageMatch = text.match(/<!--\s*image:\s*(.*?)\s*-->/);
      const title = titleMatch ? titleMatch[1].trim() : 'Untitled';

      const lines = text.split('\n');
      const contentLines = lines.filter(line => !line.trim().startsWith('#') && !line.trim().startsWith('<!--'));
      const summaryLines = contentLines.slice(0, 12).join(' ');
      const summaryHTML = marked.parseInline(summaryLines);

      const postSlug = post.replace('.md', '');
      const imageURL = imageMatch ? imageMatch[1].trim() : '';
      const imageHTML = imageURL
        ? `<img src="${imageURL}" alt="Thumbnail" class="blog-thumbnail" />`
        : '';

      const blogHTML = `
      <div class="blog-card">
        <div class="blog-image-container">${imageHTML}</div>
        <div class="blog-content">
          <h3><a href="blog-template.html?post=${postSlug}">${title}</a></h3>
          <p class="blog-summary-container">
            ${summaryHTML}<a href="blog-template.html?post=${postSlug}" class="blog-see-more"> See more</a>
          </p>
        </div>
      </div>
    `;
      

      blogList.innerHTML += blogHTML;
    });
});
