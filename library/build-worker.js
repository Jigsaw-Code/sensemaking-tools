const esbuild = require('esbuild');

async function build() {
  try {
    const result = await esbuild.build({
      entryPoints: ['index.ts'],
      bundle: true,
      outdir: 'dist',
      format: 'esm',
      target: 'es2022',
      platform: 'browser',
      external: [
        '@cloudflare/workers-types',
        // Exclude all Google Cloud dependencies
        '@google-cloud/vertexai',
        'google-auth-library',
        'gaxios',
        'jwa',
        'jws',
        'gtoken',
        'gcp-metadata',
        'google-logging-utils',
        '@tensorflow/tfjs',
        '@tensorflow/tfjs-node-gpu'
      ],
      define: {
        'process.env.NODE_ENV': '"production"',
        'process.env.TFJS_NODE_GPU': 'false'
      },
      loader: {
        '.ts': 'ts'
      },
      tsconfig: 'tsconfig.worker.json',
      minify: false,
      sourcemap: true,
      metafile: true,
      banner: {
        js: '// Built for Cloudflare Workers - OpenRouter only\n'
      },
      plugins: [
        {
          name: 'exclude-node-modules',
          setup(build) {
            build.onResolve({ filter: /^node:/ }, () => ({ external: true }));
            build.onResolve({ filter: /^(fs|path|os|crypto|stream|util|querystring|http|https|net|tls|child_process|assert)$/ }, () => ({ external: true }));
          }
        }
      ]
    });

    console.log('Build completed successfully!');
    console.log('Output files:', result.outputFiles?.map(f => f.path) || []);
  } catch (error) {
    console.error('Build failed:', error);
    process.exit(1);
  }
}

build();
