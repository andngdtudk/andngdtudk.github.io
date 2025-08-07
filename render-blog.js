const posts = [
  'control-theory.md',
  'operations-research.md',
  // Add more blog files here
];

const blogList = document.getElementById('blog-list');

Promise.all(posts.map(post =>
  fetch(`/blog-posts/${post}`).then(res => res.text())
)).then(allTexts => {
  let allBlogHTML = '';

  allTexts.forEach((text, i) => {
    const post = posts[i];

    const titleMatch = text.match(/^#\s+(.+)/m);
    const imageMatch = text.match(/<!--\s*image:\s*(.*?)\s*-->/);
    const title = titleMatch ? titleMatch[1].trim() : 'Untitled';

    const lines = text.split('\n');
    const contentLines = lines.filter(line => !line.trim().startsWith('#') && !line.trim().startsWith('<!--'));
    const summaryLines = contentLines.slice(0, 12).join(' ').trim();

    // ✅ Apply drop-cap to first letter
    const firstLetter = summaryLines.charAt(0);
    const rest = summaryLines.slice(1);
    const summaryWithDropCap = `<span class="drop-cap">${firstLetter}</span>${rest}`;
    const summaryHTML = marked.parseInline(summaryWithDropCap);

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
          <div class="blog-summary-container">${summaryHTML}</div>
          <a href="blog-template.html?post=${postSlug}" class="blog-see-more">Continue reading</a>
        </div>
      </div>
    `;

    allBlogHTML += blogHTML;
  });

  blogList.innerHTML = allBlogHTML;
});
