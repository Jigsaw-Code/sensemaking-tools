/**
 * Returns global CSS styles for tooltips and related components.
 * Defines styles for tooltip positioning, appearance, and content formatting.
 *
 * @returns {string} CSS styles as a template literal
 */
export function globalStyle() {
  return `
/* Base tooltip styles */
.sm-tooltip {
    position: fixed;
    pointer-events: none;
    opacity: 0;
    background-color: #FFF;
    box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.15), 0px 1px 2px rgba(0, 0, 0, 0.3);
    max-width: 250px;
    padding: 12px;
    font-family: var(--sm-font-family, sans-serif);
    font-size: 12px;
    z-index: 9999;
    transition: opacity 0.2s ease-in-out;
    top: -100px;
    border-radius: 16px;
}

/* Tooltip position variants */
.for-statement.sm-tooltip.bottom-left {
    border-radius: 16px 16px 16px 0;
}

.for-statement.sm-tooltip.top-left {
    border-radius: 0px 16px 16px 16px;
}

.for-statement.sm-tooltip.bottom-right {
    border-radius: 16px 16px 0px 16px;
}

.for-statement.sm-tooltip.top-right {
    border-radius: 16px 0px 16px 16px;
}

/* Inverted tooltip theme */
.sm-tooltip.is-invert {
    background: #333;
    color: #fff;
    border-radius: 8px;
}

/* Topic and subtopic styles */
.sm-tooltip-topic {
    color: #666;
    margin: 8px 0;
}

.sm-tooltip-subtopic {
    color: var(--sm-color-primary, #1E2656);
    font-weight: 400;
    margin: 8px 0;
    font-size: 14px;
}

/* Status indicator styles */
.sm-tooltip-status {
    color: #1f1f1f;
    font-size: 10px;
    line-height: 1.2;
    padding: 2px 7px;
    border-radius: 500px;
    margin: 8px 0;
    display: inline-block;
    width: fit-content;
    background: #DEDEDE;
}

.sm-tooltip-status.agree {
    background: #A5EFBA;
}

.sm-tooltip-status.disagree {
    background: #FF8983;
}

.sm-tooltip-status.pass {
    border: 1px solid #CACACA;
    background: #fff;
}

/* Comment and topics styles */
.sm-tooltip-comment {
    margin: 8px 0;
    font-size: 12px;
    color: #333;
}

.sm-tooltip-topics {
    margin: 8px 0;
    font-size: 11px;
    color: var(--sm-color-primary, #1E2656);
    font-weight: 600;
}

/* Divider */
.sm-tooltip hr {
    border-top: 1px solid #DDE1EB;
}

/* Vote display styles */
.sm-tooltip-votes div {
    margin: 8px 0;
    font-size: 11px;
    color: #333;
}

.sm-tooltip-votes ul {
    padding-left: 16px;
    margin: 0;
    list-style-type: none;
}

.sm-tooltip-votes li {
    margin: 8px 0;
    font-size: 11px;
    color: #333;
    line-height: 1;
    position: relative;
}

/* Vote indicator dots */
.sm-tooltip-votes li:before {
    content: "";
    position: absolute;
    left: 1px;
    top: 50%;
    display: block;
    font-size: 16px;
    border-radius: 500px;
    transform: translate(-100%, -50%);
    height: 7px;
    width: 7px;
}

.sm-tooltip-votes li:nth-of-type(1):before {
    background-color: #6DD58C;
}

.sm-tooltip-votes li:nth-of-type(2):before {
    background-color: #FF8983;
}

.sm-tooltip-votes li:nth-of-type(3):before {
    background-color: #fff;
    outline: 1px solid #999;
    height: 5px;
    width: 5px;
    left: 0;
}

/* Vote row layout */
.sm-tooltip-votes li span.row {
    display: flex;
    padding-left: 8px;
    white-space: nowrap;
}

.sm-tooltip-votes .count {
    text-align: right;
}

.sm-tooltip-votes span.row > span:nth-of-type(1) {
    width: 100px;
    white-space: nowrap;
}

.sm-tooltip-votes span.row > span:nth-of-type(2) {
    width: 40px;
    white-space: nowrap;
}

.sm-tooltip-votes li span:nth-of-type(2) span:nth-of-type(1) {
    font-weight: 600;
}

/* Subtopic summary styles */
.sm-tooltip-subtopic-summary {
    margin-top: 1rem;
    color: #1f1f1f;
}

.sm-tooltip-subtopic-summary ul {
    padding-left: 8px;
}

.sm-tooltip-subtopic-summary ul li {
    margin: 0.5rem;
    font-weight: 400;
}
`;
}
