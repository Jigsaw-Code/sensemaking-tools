import{s as k}from"./scriptWarning-DXObLxWB.js";const F="/sensemaking-tools/visualization-docs",d=["#AFB42B","#129EAF","#F4511E","#3949AB","#5F8F35","#9334E6","#E52592","#00897B","#E8710A","#1A73E8"],c=t=>t.startsWith("http://")||t.startsWith("https://")?t:`${F}${t}`,E={title:"Charts/TopicsDistribution",tags:["autodocs"],argTypes:{dataSource:{name:"data-source",control:"text",description:"Local path or remote URL to the data source JSON.",table:{type:{summary:"string"},defaultValue:{summary:"none"},category:"Required"}},summarySource:{name:"summary-source",control:"text",description:"Local path or remote URL to the summary data JSON. Optional, but required for theme summaries.",table:{type:{summary:"string"},defaultValue:{summary:"none"},category:"Optional"}},view:{control:"select",options:["cluster","scatter"],description:'Display mode: "cluster" (circle packing) or "scatter" (distributed). Can be set statically or dynamically via DOM manipulation.',table:{type:{summary:"string"},defaultValue:{summary:"cluster"}}},id:{control:"text",description:"Unique identifier for the chart element. Primarily used to target the chart for DOM manipulation.",table:{type:{summary:"string"},defaultValue:{summary:"none"}}},topicFilter:{name:"topic-filter",control:"text",description:"Semicolon-separated list of topics to filter data. Can also prefix with '!' to exclude topics.",table:{type:{summary:"string"},defaultValue:{summary:"none"},category:"Required"}},colors:{name:"colors",control:"object",description:"Array of colors to use in the chart.",table:{type:{summary:"string[]"},defaultValue:{summary:JSON.stringify(d)},category:"Style"}},fontFamily:{name:"font-family",control:"text",description:"Font family to use in the chart.",table:{type:{summary:"string"},defaultValue:{summary:"Noto Sans"},category:"Style"}},altText:{name:"alt-text",control:"text",description:"Manually set alternative text description for accessibility purposes. This will overwrite the programmatically generated alt text.",table:{type:{summary:"string"},defaultValue:{summary:"undefined"},category:"Accessibility"}}},parameters:{docs:{description:{component:`
The topic alignment chart displays agreement/disagreement percentages with options for different view options (cluster or scatter). The view can be set statically or dynamically via DOM manipulation.

The radius of each subtopic is determined by the number of statements in the subtopic using a square root scale.
                
${k}
                `}}}},x=({id:t,dataSource:l,summarySource:s,view:u,topicFilter:o,colors:i,fontFamily:n,altText:m})=>`
    <sensemaker-chart
      id="${t}"
      data-source="${c(l)}"
      summary-source="${c(s)}"
      chart-type="topics-distribution"
      view="${u}"
      colors='${JSON.stringify(i||d)}'
      font-family="${n||"Noto Sans"}"
      ${o?`topic-filter="${o}"`:""}
      ${m?`alt-text="${m}"`:""}
    ></sensemaker-chart>
  `,e=x.bind({});e.args={dataSource:"/comments.json",summarySource:"/summary.json",view:"cluster",topicFilter:"!other"};e.parameters={docs:{description:{story:"The cluster view shows the subtopics as circles grouped by topic, with the size of the circle indicating the number of statements in the subtopic."},source:{code:`<sensemaker-chart
  data-source="/comments.json"
  summary-source="/summary.json"
  chart-type="topics-distribution"
  view="cluster"
  topic-filter="!other">
</sensemaker-chart>`,language:"html",type:"code"}}};const r=x.bind({});r.args={dataSource:"/comments.json",summarySource:"/summary.json",view:"scatter",topicFilter:"!other"};r.parameters={docs:{description:{story:"The scatter view shows the subtopics as circles distributed by topic and alignment rate, with the size of the circle indicating the number of statements in the subtopic."},source:{code:`<sensemaker-chart
  data-source="/comments.json"
  summary-source="/summary.json"
  chart-type="topics-distribution"
  view="scatter"
  topic-filter="!other">
</sensemaker-chart>`,language:"html",type:"code"}}};const V=({dataSource:t,summarySource:l,topicFilter:s,colors:u,fontFamily:o,altText:i})=>(setTimeout(()=>{const n=document.getElementById("topics-distribution-chart-with-toggle");document.querySelectorAll('input[name="view"]').forEach(T=>{T.addEventListener("change",p=>{p.target.checked&&(p.target.value==="scatter"?n.setAttribute("view","scatter"):n.setAttribute("view","cluster"))})})},100),`
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
      <div class="view-controls" style="margin-bottom: 20px;">
        <label style="margin-right: 15px; cursor: pointer;">
          <input type="radio" name="view" value="cluster" checked /> Cluster View
        </label>
        <label style="cursor: pointer;">
          <input type="radio" name="view" value="scatter" /> Scatter View
        </label>
      </div>
      
      <sensemaker-chart
        id="topics-distribution-chart-with-toggle"
        data-source="${c(t)}"
        summary-source="${c(l)}"
        chart-type="topics-distribution"
        view="cluster"
        colors='${JSON.stringify(u||d)}'
        font-family="${o||"Noto Sans"}"
        ${s?`topic-filter="${s}"`:""}
        ${i?`alt-text="${i}"`:""}
      ></sensemaker-chart>
    </div>
  `),a=V.bind({});a.args={dataSource:"/comments.json",summarySource:"/summary.json",topicFilter:"!other"};a.parameters={docs:{description:{story:"The chart view updates via external controls, with animated transitions that preserve DOM state."},source:{code:`<!-- Toggle controls -->
<div class="view-controls">
  <label>
    <input type="radio" name="view" value="cluster" checked /> cluster View
  </label>
  <label>
    <input type="radio" name="view" value="scatter" /> scatter View
  </label>
</div>

<!-- Chart component -->
<sensemaker-chart
  id="topics-distribution-chart-with-toggle"
  data-source="/comments.json"
  summary-source="/summary.json"
  chart-type="topics-distribution"
  view="cluster"
  topic-filter="!other">
</sensemaker-chart>

<script>
  // Add event listeners to view controls
  const chart = document.getElementById("alignment-chart-with-toggle");
  const viewInputs = document.querySelectorAll('input[name="view"]');
  
  viewInputs.forEach((input) => {
    input.addEventListener("change", (e) => {
      if (e.target.checked) {
        chart.setAttribute("view", e.target.value);
      }
    });
  });
<\/script>`,language:"html",type:"code"}}};var y,h,g;e.parameters={...e.parameters,docs:{...(y=e.parameters)==null?void 0:y.docs,source:{originalSource:`({
  id,
  dataSource,
  summarySource,
  view,
  topicFilter,
  colors,
  fontFamily,
  altText
}) => {
  return \`
    <sensemaker-chart
      id="\${id}"
      data-source="\${getDataSource(dataSource)}"
      summary-source="\${getDataSource(summarySource)}"
      chart-type="topics-distribution"
      view="\${view}"
      colors='\${JSON.stringify(colors || defaultColors)}'
      font-family="\${fontFamily || 'Noto Sans'}"
      \${topicFilter ? \`topic-filter="\${topicFilter}"\` : ""}
      \${altText ? \`alt-text="\${altText}"\` : ""}
    ></sensemaker-chart>
  \`;
}`,...(g=(h=e.parameters)==null?void 0:h.docs)==null?void 0:g.source}}};var f,v,w;r.parameters={...r.parameters,docs:{...(f=r.parameters)==null?void 0:f.docs,source:{originalSource:`({
  id,
  dataSource,
  summarySource,
  view,
  topicFilter,
  colors,
  fontFamily,
  altText
}) => {
  return \`
    <sensemaker-chart
      id="\${id}"
      data-source="\${getDataSource(dataSource)}"
      summary-source="\${getDataSource(summarySource)}"
      chart-type="topics-distribution"
      view="\${view}"
      colors='\${JSON.stringify(colors || defaultColors)}'
      font-family="\${fontFamily || 'Noto Sans'}"
      \${topicFilter ? \`topic-filter="\${topicFilter}"\` : ""}
      \${altText ? \`alt-text="\${altText}"\` : ""}
    ></sensemaker-chart>
  \`;
}`,...(w=(v=r.parameters)==null?void 0:v.docs)==null?void 0:w.source}}};var b,S,$;a.parameters={...a.parameters,docs:{...(b=a.parameters)==null?void 0:b.docs,source:{originalSource:`({
  dataSource,
  summarySource,
  topicFilter,
  colors,
  fontFamily,
  altText
}) => {
  // This will run after the component is added to the DOM
  setTimeout(() => {
    const chart = document.getElementById("topics-distribution-chart-with-toggle");
    const viewInputs = document.querySelectorAll('input[name="view"]');

    // Add event listeners to view controls
    viewInputs.forEach(input => {
      input.addEventListener("change", e => {
        if (e.target.checked) {
          if (e.target.value === "scatter") {
            chart.setAttribute("view", "scatter");
          } else {
            chart.setAttribute("view", "cluster");
          }
        }
      });
    });
  }, 100);
  return \`
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
      <div class="view-controls" style="margin-bottom: 20px;">
        <label style="margin-right: 15px; cursor: pointer;">
          <input type="radio" name="view" value="cluster" checked /> Cluster View
        </label>
        <label style="cursor: pointer;">
          <input type="radio" name="view" value="scatter" /> Scatter View
        </label>
      </div>
      
      <sensemaker-chart
        id="topics-distribution-chart-with-toggle"
        data-source="\${getDataSource(dataSource)}"
        summary-source="\${getDataSource(summarySource)}"
        chart-type="topics-distribution"
        view="cluster"
        colors='\${JSON.stringify(colors || defaultColors)}'
        font-family="\${fontFamily || 'Noto Sans'}"
        \${topicFilter ? \`topic-filter="\${topicFilter}"\` : ""}
        \${altText ? \`alt-text="\${altText}"\` : ""}
      ></sensemaker-chart>
    </div>
  \`;
}`,...($=(S=a.parameters)==null?void 0:S.docs)==null?void 0:$.source}}};const O=["clusterView","scatterView","WithViewToggle"];export{a as WithViewToggle,O as __namedExportsOrder,e as clusterView,E as default,r as scatterView};
