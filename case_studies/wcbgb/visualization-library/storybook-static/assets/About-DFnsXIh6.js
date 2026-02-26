import{j as e,c as a}from"./index-DLD6X-TB.js";import{useMDXComponents as t}from"./index-BlS-vKiI.js";import"./iframe-B2RltiIt.js";import"./index-COZDKLvM.js";import"./index-CfOrKyLd.js";import"./index-DrFu-skq.js";const o="/sensemaking-tools/visualization-docs/assets/topics-alignment-BuuLHVvW.png",r="/sensemaking-tools/visualization-docs/assets/topics-overview-DYReovpg.png",l="/sensemaking-tools/visualization-docs/assets/topics-distribution-BPKohf-q.png",x=()=>{const s={path:"path",svg:"svg",...t()};return e.jsx(s.svg,{viewBox:"0 0 14 14",width:"8px",height:"14px",style:{marginLeft:"4px",display:"inline-block",shapeRendering:"inherit",verticalAlign:"middle",fill:"currentColor","path fill":"currentColor"},children:e.jsx(s.path,{d:"m11.1 7.35-5.5 5.5a.5.5 0 0 1-.7-.7L10.04 7 4.9 1.85a.5.5 0 1 1 .7-.7l5.5 5.5c.2.2.2.5 0 .7Z"})})};function n(s){const i={a:"a",code:"code",h1:"h1",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",...t(),...s.components};return e.jsxs(e.Fragment,{children:[e.jsx(a,{title:"Using the library"}),`
`,e.jsxs("div",{className:"sb-container",children:[e.jsxs("div",{className:"sb-section-title",children:[e.jsx(i.h1,{id:"sensemaker-visualization-library",children:"Sensemaker Visualization Library"}),e.jsxs(i.p,{children:["A set of web components for visualizing conversation data processed by the ",e.jsx(i.a,{href:"https://jigsaw-code.github.io/sensemaking-tools/",rel:"nofollow",children:"Sensemaker"})," tool."]})]}),e.jsxs("div",{className:"sb-section",children:[e.jsxs("div",{className:"sb-section-item",children:[e.jsx("img",{src:o,alt:"Topic Alignment Chart visualization"}),e.jsx("h4",{className:"sb-section-item-heading",children:"Topic Alignment Chart"}),e.jsx("p",{className:"sb-section-item-paragraph",children:"Displays the alignment of all statements in a given topic."})]}),e.jsxs("div",{className:"sb-section-item",children:[e.jsx("img",{src:r,alt:"Topics Overview visualization"}),e.jsx("h4",{className:"sb-section-item-heading",children:"Topics Overview"}),e.jsx("p",{className:"sb-section-item-paragraph",children:"Displays topic distribution and statement counts using stacked bar charts."})]}),e.jsxs("div",{className:"sb-section-item",children:[e.jsx("img",{src:l,alt:"Topics Distribution visualization"}),e.jsxs("div",{children:[e.jsx("h4",{className:"sb-section-item-heading",children:"Topics Distribution"}),e.jsx("p",{className:"sb-section-item-paragraph",children:"Displays topic and subtopic statement count and aggregated statement alignment."})]})]})]})]}),`
`,e.jsx("div",{className:"sb-container",children:e.jsxs("div",{className:"sb-section-title",children:[e.jsx(i.h1,{id:"implementation-steps",children:"Implementation Steps"}),e.jsxs(i.ol,{children:[`
`,e.jsxs(i.li,{children:[e.jsx(i.strong,{children:"Add the library"}),": Include the library via CDN or npm"]}),`
`]}),e.jsx(i.p,{children:e.jsx(i.strong,{children:"Using npm:"})}),e.jsx(i.pre,{children:e.jsx(i.code,{className:"language-bash",children:`npm i @conversationai/sensemaker-visualizations
`})}),e.jsx(i.p,{children:"Then import in your JavaScript/TypeScript:"}),e.jsx(i.pre,{children:e.jsx(i.code,{className:"language-javascript",children:`import '@conversationai/sensemaker-visualizations';
`})}),e.jsx(i.p,{children:e.jsx(i.strong,{children:"Using CDN:"})}),e.jsx(i.pre,{children:e.jsx(i.code,{className:"language-html",children:`<script type="module" src="https://cdn.jsdelivr.net/npm/@conversationai/sensemaker-visualizations@latest/dist/sensemaker-chart.umd.js"><\/script>
`})}),e.jsxs(i.ol,{start:"2",children:[`
`,e.jsxs(i.li,{children:[e.jsx(i.strong,{children:"Add Components"}),": Place the web components in your HTML"]}),`
`,e.jsxs(i.li,{children:[e.jsx(i.strong,{children:"Configure"}),": Set attributes for data source, view type, and filters"]}),`
`,e.jsxs(i.li,{children:[e.jsx(i.strong,{children:"Style"}),": Customize appearance using CSS variables"]}),`
`]}),e.jsx(i.p,{children:"See the sidebar for detailed component documentation and configuration options."})]})}),`
`,e.jsxs("div",{className:"sb-container",children:[e.jsx("div",{className:"sb-section-title",children:e.jsx(i.h1,{id:"example-usage",children:"Example Usage"})}),e.jsx(i.pre,{children:e.jsx(i.code,{className:"language-html",children:`
<sensemaker-chart
  data-source="./comments.json"
  chart-type="topic-alignment"
  view="solid"
  topic-filter="education">
</sensemaker-chart>

<script type="module" src="https://cdn.jsdelivr.net/npm/sensemaker-viz-components@latest/dist/sensemaker-chart.umd.js"><\/script>
`})})]}),`
`,e.jsxs("div",{className:"sb-container",children:[e.jsx("div",{className:"sb-section-title",children:e.jsx(i.h1,{id:"data-source-and-license",children:"Data Source and License"})}),e.jsxs(i.p,{children:["The data used in this demo was gathered using the ",e.jsx(i.a,{href:"https://compdemocracy.org/Polis/",rel:"nofollow",children:"Polis software"})," and is sub-licensed under CC BY 4.0 with Attribution to The Computational Democracy Project. The data and more information about how the data was collected can be found at the following link:"]}),e.jsx(i.p,{children:e.jsx(i.a,{href:"https://github.com/compdemocracy/openData/tree/master/american-assembly.bowling-green",rel:"nofollow",children:"https://github.com/compdemocracy/openData/tree/master/american-assembly.bowling-green"})})]}),`
`,e.jsx("style",{children:`
  .sb-container {
    margin-bottom: 48px;
  }

  .sb-section {
    width: 100%;
    display: flex;
    flex-direction: row;
    gap: 20px;
  }

  .sb-section-item {
    flex: 1;
    display: flex;
    flex-direction: column;
  }

  .sb-section-item img {
    width: 100%;
    aspect-ratio: 4/3;
    object-fit: cover;
    border-radius: 4px;
    object-position: top left;
    border: 1px solid #E0E0E0;
  }

  .sb-section-title {
    margin-bottom: 32px;
  }

  .sb-section a:not(h1 a, h2 a, h3 a) {
    font-size: 14px;
  }

  .sb-section-item-heading {
    padding-top: 20px !important;
    padding-bottom: 5px !important;
    margin: 0 !important;
  }

  .sb-section-item-paragraph {
    margin: 0;
    padding-bottom: 10px;
  }

  .sb-chevron {
    margin-left: 5px;
  }

  .sb-features-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    grid-gap: 32px 20px;
  }

  .sb-socials {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
  }

  .sb-socials p {
    margin-bottom: 10px;
  }

  .sb-explore-image {
    max-height: 32px;
    align-self: flex-start;
  }

  .sb-addon {
    width: 100%;
    display: flex;
    align-items: center;
    position: relative;
    background-color: #EEF3F8;
    border-radius: 5px;
    border: 1px solid rgba(0, 0, 0, 0.05);
    background: #EEF3F8;
    height: 180px;
    margin-bottom: 48px;
    overflow: hidden;
  }

  .sb-addon-text {
    padding-left: 48px;
    max-width: 240px;
  }

  .sb-addon-text h4 {
    padding-top: 0px;
  }

  .sb-addon-img {
    position: absolute;
    left: 345px;
    top: 0;
    height: 100%;
    width: 200%;
    overflow: hidden;
  }

  .sb-addon-img img {
    width: 650px;
    transform: rotate(-15deg);
    margin-left: 40px;
    margin-top: -72px;
    box-shadow: 0 0 1px rgba(255, 255, 255, 0);
    backface-visibility: hidden;
  }

  @media screen and (max-width: 800px) {
    .sb-addon-img {
      left: 300px;
    }
  }

  @media screen and (max-width: 600px) {
    .sb-section {
      flex-direction: column;
    }

    .sb-features-grid {
      grid-template-columns: repeat(1, 1fr);
    }

    .sb-socials {
      grid-template-columns: repeat(2, 1fr);
    }

    .sb-addon {
      height: 280px;
      align-items: flex-start;
      padding-top: 32px;
      overflow: hidden;
    }

    .sb-addon-text {
      padding-left: 24px;
    }

    .sb-addon-img {
      right: 0;
      left: 0;
      top: 130px;
      bottom: 0;
      overflow: hidden;
      height: auto;
      width: 124%;
    }

    .sb-addon-img img {
      width: 1200px;
      transform: rotate(-12deg);
      margin-left: 0;
      margin-top: 48px;
      margin-bottom: -40px;
      margin-left: -24px;
    }
  }
  `})]})}function b(s={}){const{wrapper:i}={...t(),...s.components};return i?e.jsx(i,{...s,children:e.jsx(n,{...s})}):n(s)}export{x as RightArrow,b as default};
