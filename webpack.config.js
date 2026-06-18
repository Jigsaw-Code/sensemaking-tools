/* globals __dirname, module, require */

const CopyPlugin = require('copy-webpack-plugin');
const HtmlInjectAttrsPlugin = require('html-webpack-inject-attributes-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const StylelintPlugin = require('stylelint-webpack-plugin');
const TerserPlugin = require('terser-webpack-plugin');
const path = require('path');

const src = './src';
const dest = './site';

module.exports = (env, argv) => {
  const config = {
    entry: {
      app: path.resolve(__dirname, `${src}/app/app.ts`),
      core_init: path.resolve(__dirname, `${src}/app/core/core-init.ts`),
      core_defer: path.resolve(__dirname, `${src}/app/core/core-defer.ts`),
    },
  };

  if (argv.mode === 'development') {
    config.devtool = 'inline-source-map';
  }

  // build options
  let minify = false;
  let sourceMap = true;
  if (argv.mode === 'production') {
    minify = true;
    sourceMap = false;
  }

  // Generate excluded entry points
  const excludedChunks = [];
  for (const key in config.entry) {
    if (key !== 'app') {
      excludedChunks.push(key);
    }
  }

  config.module = {
    rules: [
      {
        test: /\.tsx?$/,
        loader: 'ts-loader',
        options: {
          configFile: require.resolve('./tsconfig.json'),
        },
      },

      {
        test: /\.scss$/i,
        use: [
          MiniCssExtractPlugin.loader,
          {
            loader: 'css-loader',
            options: {
              importLoaders: 1,
            },
          },
          {
            loader: 'postcss-loader',
            options: {
              postcssOptions: {
                plugins: [
                  [
                    'postcss-preset-env',
                    {
                      features: {
                        'logical-properties-and-values': false,
                      },
                    },
                  ],
                ],
                sourceMap,
              },
            },
          },
          {
            loader: 'resolve-url-loader',
            options: {
              sourceMap,
            },
          },
          {
            loader: 'sass-loader',
            options: {
              sassOptions: {
                quietDeps: true,
              },
              sourceMap: true, // required for resolve-url-loader to work, even though we don't want them in the build
            },
          },
        ],
      },

      {
        test: /\.(png|svg|jpe?g|gif)$/i,
        type: 'asset/resource',
        generator: {
          filename: 'assets/images/[path][name].[ext]',
        },
      },

      {
        test: /\.(woff|woff2|eot|ttf|otf)$/i,
        type: 'asset/resource',
        generator: {
          filename: 'assets/fonts/[path][name].[ext]',
        },
      },

      {
        test: /\.html$/i,
        loader: 'html-loader',
        exclude: [path.resolve(__dirname, `${src}/index.html`)],
      },
    ],
  };

  config.plugins = [
    new MiniCssExtractPlugin({
      filename: 'styles-preload.css',
    }),

    new MiniCssExtractPlugin({
      filename: 'styles.css',
    }),

    // new StylelintPlugin({
    //   context: src,
    //   files: ['styles/**/*.scss'],
    // }),

    new HtmlWebpackPlugin({
      inject: 'body',
      template: path.resolve(__dirname, `${src}/index.html`),
      excludeChunks: excludedChunks,
      minify,
      attributes: {
        rel: tag => {
          if (tag.tagName === 'link' && tag.attributes.href.includes('styles-preload')) {
            return 'preload';
          }
          return tag.attributes.rel;
        },
        as: tag => {
          if (tag.tagName === 'link' && tag.attributes.href.includes('styles-preload')) {
            return 'style';
          }
          return tag.attributes.as;
        },
        href: tag => {
          let href = tag.attributes.href;
          if (tag.tagName === 'link') {
            if (href.includes('styles-preload')) {
              href = href.replace('-preload', '');
            }
            if (href.includes('styles')) {
              href = `${href}`;
            }
          }
          return href;
        },
        src: tag => {
          if (tag.tagName === 'script') {
            return `${tag.attributes.src}`;
          }
          return tag.attributes.src;
        },
      },
    }),

    new HtmlInjectAttrsPlugin(),

    new CopyPlugin({
      patterns: [
        {
          from: path.resolve(__dirname, `${src}/assets`),
          to: path.resolve(__dirname, `${dest}/assets`),
        },
        {
          from: path.resolve(__dirname, `${src}/robots.txt`),
          to: path.resolve(__dirname, `${dest}`),
        },
        {
          from: path.resolve(__dirname, `${src}/sitemap.xml`),
          to: path.resolve(__dirname, `${dest}`),
        },
        {
          from: path.resolve(__dirname, `${src}/manifest.json`),
          to: path.resolve(__dirname, `${dest}`),
        },
        {
          from: '_glue/icons/glue-icons.svg',
          to: path.resolve(__dirname, `${dest}/assets/images`),
        },
      ],
    }),
  ];

  if (argv.mode === 'production') {
    config.plugins.push(
      new TerserPlugin({
        terserOptions: {
          format: {
            comments: 'some',
          },
        },
        extractComments: false,
      }),
    );
  }

  config.resolve = {
    extensions: ['.ts', '...'],
  };

  config.output = {
    filename: '[name].bundle.js',
    path: path.resolve(__dirname, `${dest}`),
    clean: true,
  };

  return config;
};
