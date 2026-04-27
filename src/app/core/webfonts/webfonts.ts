module WebfontsModule {
  function init(): void {
    const link = document.createElement('link');
    link.href = 'https://fonts.googleapis.com/css2?family=Google+Sans+Text:wght@400&family=Google+Sans:ital,wght@0,400;0,500;0,700;1,400;1,500;1,700&display=swap';
    link.rel = 'stylesheet';
    document.head.appendChild(link);
  }

  init();
}
