import { shadeColor } from "../../helpers/shadeColor.js";

/**
 * @typedef {Object} Group
 * @property {string} label - Display label for the group
 * @property {string} value - Internal identifier for the group
 */

/** @type {Group[]} */
const groups = [{
    label: "Alignment",
    value: "alignment"
}, {
    label: "Uncertainty",
    value: "uncertainty"
}, {
    label: "Uncategorized",
    value: "uncategorized"
}];

/**
 * Creates a grid container for either solid or waffle view.
 * The grid organizes statements into groups based on their alignment and uncertainty scores.
 * 
 * @param {Object} params - Grid creation parameters
 * @param {string} params.type - Type of grid ('solid' or 'waffle')
 * @param {number} params.width - Width of the grid
 * @param {number} [params.height] - Height of the grid (required for solid view)
 * @param {Object} params.data - Data containing percentages for each group
 * @param {Object} params.info - Tooltip information for each group
 * @param {Object} params.labelTooltip - Tooltip instance for labels
 * @returns {HTMLElement} Grid container element
 */
function createGrid({ type = 'solid', width, height, data, info, labelTooltip }) {
    const grid = document.createElement("div");
    grid.className = `${type}-grid`;

    groups.forEach((group, groupIndex) => {
        const groupElement = document.createElement("div");
        groupElement.className = `${type}-group ${group.value}`;
        
        let percentage;
        if (group.value === "alignment") {
            percentage = parseInt(data.percentages.high) + parseInt(data.percentages.low);
        } else if (group.value === "uncategorized") {
            percentage = parseInt(data.percentages.uncategorized);
        } else if (group.value === "uncertainty") {
            percentage = parseInt(data.percentages.uncertainty);
        }

        // Set height for solid view
        if (type === 'solid') {
            const sectionHeight = height * (parseInt(percentage) / 100);
            groupElement.style.height = `${sectionHeight}px`;
            if (sectionHeight < 20) {
                groupElement.classList.add("is-short");
            }
        }

        const label = document.createElement("div");
        label.tabIndex = 0;
        label.className = "group-label";
        label.textContent = group.label;
        groupElement.appendChild(label);
        
        addHover(label, labelTooltip, info[group.value]);

        const box = document.createElement("div");
        box.className = `${type}-group-box`;

        groupElement.appendChild(box);

        if (percentage === 0) {
            groupElement.style.display = "none";
        }

        grid.appendChild(groupElement);
    });

    return grid;
}

/**
 * Creates a solid view visualization showing the distribution of statements.
 * The solid view represents each group as a vertical bar with sections for high/low alignment.
 * 
 * @param {Object} params - Solid view parameters
 * @param {number} params.width - Width of the visualization
 * @param {number} params.height - Height of the visualization
 * @param {Object} params.data - Data containing percentages and counts
 * @param {Object} params.info - Tooltip information
 * @param {Object} params.theme - Theme configuration
 * @param {Object} params.labelTooltip - Tooltip instance for labels
 * @returns {{solidViz: HTMLElement}} Solid view element
 */
export function createSolid({
    width,
    height,
    data,
    info,
    theme,
    labelTooltip
}) {
    const solidViz = document.createElement("div");
    solidViz.className = "solid-viz";

    const grid = createGrid({ type: 'solid', width, height, data, info, labelTooltip });

    /**
     * Adds text content to a section showing percentage and label
     * @param {HTMLElement} section - Section element to add text to
     * @param {number} percentage - Percentage to display
     */
    function addText(section, percentage) {
        const text = document.createElement("div");
        text.className = "solid-group-box-section-text";

        const percentageEl = document.createElement("div");
        percentageEl.className = "percentage";
        percentageEl.textContent = `${percentage}%`;
        text.appendChild(percentageEl);

        const subtitle = document.createElement("div");
        subtitle.className = "solid-group-box-section-text-subtitle";
        subtitle.textContent = "Of statements";
        text.appendChild(subtitle);

        section.appendChild(text);
    }

    const isNarrow = (percentage, sectionHeight) => width * percentage / 100 < 70 || sectionHeight < 90 ? "is-narrow" : "";
    const isHidden = (percentage, sectionHeight) => width * percentage / 100 < 40 || sectionHeight < 70 ? "is-hidden" : "";

    // Process each group's box
    grid.querySelectorAll('.solid-group-box').forEach((box, groupIndex) => {
        const group = groups[groupIndex];
        let percentage;

        if (group.value === "alignment") {
            percentage = parseInt(data.percentages.high) + parseInt(data.percentages.low);
        } else if (group.value === "uncategorized") {
            percentage = parseInt(data.percentages.uncategorized);
        } else if (group.value === "uncertainty") {
            percentage = parseInt(data.percentages.uncertainty);
        }

        const sectionHeight = height * (parseInt(percentage) / 100);

        if (group.value === "alignment") {
            // High alignment section
            const highSection = document.createElement("div");
            highSection.className = `solid-group-box-section high ${isNarrow(parseInt(data.percentages.high), sectionHeight)} ${isHidden(parseInt(data.percentages.high), sectionHeight)}`;
            const highSectionWidth = parseInt(data.percentages.high) / percentage * 100;
            highSection.style.width = `${highSectionWidth}%`;
            addText(highSection, parseInt(data.percentages.high));
            highSection.style.backgroundColor = theme.colors[0];
            highSection.style.color = "#fff";
            box.appendChild(highSection);

            const highSectionLabel = document.createElement("div");
            highSectionLabel.tabIndex = 0;
            highSectionLabel.className = "section-label high";
            highSectionLabel.textContent = "High";
            highSectionLabel.style.color = shadeColor(theme.colors[0], -20);
            highSection.appendChild(highSectionLabel);
            addHover(highSectionLabel, labelTooltip, info['high']);

            // Low alignment section
            const lowSection = document.createElement("div");
            lowSection.className = `solid-group-box-section low ${isNarrow(parseInt(data.percentages.low), sectionHeight)} ${isHidden(parseInt(data.percentages.low), sectionHeight)}`;
            const lowSectionWidth = parseInt(data.percentages.low) / percentage * 100;
            lowSection.style.width = `${lowSectionWidth}%`;
            addText(lowSection, parseInt(data.percentages.low));
            lowSection.style.backgroundColor = theme.colors[1];
            lowSection.style.color = "#fff";
            box.appendChild(lowSection);

            const lowSectionLabel = document.createElement("div");
            lowSectionLabel.tabIndex = 0;
            lowSectionLabel.className = "section-label low";
            lowSectionLabel.textContent = "Low";
            lowSectionLabel.style.color = shadeColor(theme.colors[1], -20);
            if (lowSectionWidth/100*width < 25) {
                lowSectionLabel.style.left = "-18px";
            }
            lowSection.appendChild(lowSectionLabel);
            addHover(lowSectionLabel, labelTooltip, info['low']);
        } else {
            let section = document.createElement("div");
            section.className = `solid-group-box-section ${group.value} ${isNarrow(percentage, sectionHeight)} ${isHidden(percentage, sectionHeight)}`;
            section.style.width = "100%";
            addText(section, percentage);
            box.appendChild(section);

            if (group.value === "uncertainty") {
                section.style.backgroundColor = "#fff";
                section.style.color = "#000";
            } else if (group.value === "uncategorized") {
                section.style.backgroundColor = theme.colors[3];
                section.style.color = "#fff";
            }
        }
    });

    solidViz.appendChild(grid);

    return { solidViz };
}

/**
 * Creates a waffle view visualization showing individual statements.
 * The waffle view represents each statement as a square in a grid layout.
 * 
 * @param {Object} params - Waffle view parameters
 * @param {number} params.width - Width of the visualization
 * @param {Object} params.data - Data containing percentages and items
 * @param {Object} params.info - Tooltip information
 * @param {Object} params.theme - Theme configuration
 * @param {Object} params.labelTooltip - Tooltip instance for labels
 * @param {Object} params.vizTooltip - Tooltip instance for visualization elements
 * @returns {{waffleViz: HTMLElement}} Waffle view element
 */
export function createWaffle({
    width,
    data,
    info,
    theme,
    labelTooltip,
    vizTooltip
}) {
    const squareSize = 25;

    const waffleViz = document.createElement("div");
    waffleViz.className = "waffle-viz";

    if (!width) return { waffleViz };

    const grid = createGrid({ type: 'waffle', width, data, info, labelTooltip });

    // Process each group's box
    grid.querySelectorAll('.waffle-group-box').forEach((box, groupIndex) => {
        const group = groups[groupIndex];
        let percentage;

        if (group.value === "alignment") {
            percentage = parseInt(data.percentages.high) + parseInt(data.percentages.low);
        } else if (group.value === "uncategorized") {
            percentage = parseInt(data.percentages.uncategorized);
        } else if (group.value === "uncertainty") {
            percentage = parseInt(data.percentages.uncertainty);
        }

        if (group.value === "alignment") {
            const leftSideGap = 10;
            const rightSideOffset = (width - 100 - leftSideGap) % squareSize;
            const labelWidth = 100;

            const plotWidth = width - labelWidth - leftSideGap;

            const largerSection = parseInt(data.percentages.high) > parseInt(data.percentages.low) ? "high" : "low";
            const largerSectionPercentage = largerSection === "high" ? parseInt(data.percentages.high) : parseInt(data.percentages.low);
            const largerSectionColumns = Math.floor(largerSectionPercentage / percentage * (plotWidth) / squareSize);
            let largerSectionWidth = (largerSectionColumns * squareSize) - squareSize;
           
            const smallerSectionColumns = Math.floor((plotWidth - largerSectionWidth - squareSize) / squareSize);
            let smallerSectionWidth = smallerSectionColumns * squareSize;

            if (!smallerSectionWidth) {
                largerSectionWidth -= squareSize;
                smallerSectionWidth = squareSize;
            }

            const highSection = document.createElement("div");
            highSection.className = `waffle-group-box-section high`;
            highSection.style.width = `${largerSection == "high" ? largerSectionWidth : smallerSectionWidth}px`;

            const highSectionLabel = document.createElement("div");
            highSectionLabel.className = "section-label high";
            highSectionLabel.textContent = "High";
            highSectionLabel.style.color = shadeColor(theme.colors[0], -20);
            highSection.appendChild(highSectionLabel);
            highSectionLabel.tabIndex = 0;
            addHover(highSectionLabel, labelTooltip, info['high']);

            let highCount = data.high.length;
            for (let i = 0; i < highCount; i++) {
                addSquare({ el: highSection, data: data.high[i], fill: theme.colors[0], squareSize, vizTooltip });
            }

            box.appendChild(highSection);

            const lowSection = document.createElement("div");
            lowSection.className = `waffle-group-box-section low`;
            lowSection.style.width = `${largerSection == "low" ? largerSectionWidth : smallerSectionWidth}px`;
            lowSection.style.marginRight = `${rightSideOffset}px`;

            const lowSectionLabel = document.createElement("div");
            lowSectionLabel.className = "section-label low";
            lowSectionLabel.textContent = "Low";
            lowSectionLabel.style.color = shadeColor(theme.colors[1], -20);
            lowSection.appendChild(lowSectionLabel);
            lowSectionLabel.tabIndex = 0;
            addHover(lowSectionLabel, labelTooltip, info['low']);

            let lowCount = data.low.length;
            for (let i = 0; i < lowCount; i++) {
                addSquare({ el: lowSection, data: data.low[i], fill: theme.colors[1], squareSize, vizTooltip });
            }
            box.appendChild(lowSection);
        } else {
            let section = document.createElement("div");
            section.className = `waffle-group-box-section ${group.value}`;
            section.style.width = "100%";

            let count = data[group.value].length;
            box.appendChild(section);

            if (group.value === "uncertainty") {
                for (let i = 0; i < count; i++) {
                    addSquare({ el: section, data: data[group.value][i], fill: "#fff", squareSize, invert: true, vizTooltip });
                }
            } else if (group.value === "uncategorized") {
                for (let i = 0; i < count; i++) {
                    addSquare({ el: section, data: data[group.value][i], fill: theme.colors[3], squareSize, vizTooltip });
                }
            }
        }
    });

    waffleViz.appendChild(grid);

    return { waffleViz };
}

/**
 * Calculates the total number of votes for a statement
 * @param {Object} votes - Vote counts object
 * @param {string} [type='total'] - Type of votes to count ('agree', 'disagree', 'pass', or 'total')
 * @returns {number} Total number of votes
 */
function calculateVotes(votes, type = 'total') {
    if (!votes || typeof votes !== "object") {
        return 0;
    }

    return Object.values(votes).reduce((total, groupVotes) => {
        switch (type) {
            case 'agree':
                return total + groupVotes.agreeCount;
            case 'disagree':
                return total + groupVotes.disagreeCount;
            case 'pass':
                return total + groupVotes.passCount;
            case 'total':
            default:
                return total + groupVotes.agreeCount + groupVotes.disagreeCount + groupVotes.passCount;
        }
    }, 0);
}

/**
 * Adds hover and focus event handlers to show tooltips
 * @param {HTMLElement} el - Element to add tooltip to
 * @param {Object} tooltip - Tooltip instance
 * @param {string} info - Tooltip content
 */
function addHover(el, tooltip, info) {
    el.addEventListener("mouseenter", (e) => {
        tooltip.show(info, e.clientX, e.clientY, "is-invert");
    });

    el.addEventListener("mousemove", (e) => {
        tooltip.move(e.clientX, e.clientY);
    });

    el.addEventListener("mouseleave", () => {
        tooltip.hide("is-invert");
    });

    el.addEventListener("focus", (e) => {
        const rect = el.getBoundingClientRect();
        tooltip.show(info, rect.left + rect.width / 2, rect.top, "is-invert");
    });
    
    el.addEventListener("blur", () => {
        tooltip.hide("is-invert");
    });
}

/**
 * Adds a square to the waffle visualization representing a statement
 * @param {Object} params - Square parameters
 * @param {HTMLElement} params.el - Parent element to add square to
 * @param {Object} params.data - Statement data
 * @param {string} params.fill - Background color
 * @param {number} params.squareSize - Size of the square
 * @param {boolean} [params.invert=false] - Whether to invert the square's appearance
 * @param {Object} params.vizTooltip - Tooltip instance for the square
 */
function addSquare({ el, data, fill, squareSize, invert = false, vizTooltip }) {
    const square = document.createElement("div");
    square.className = "waffle-square";
    if (invert) {
        square.className += " invert";
    }
    square.style.width = `${squareSize}px`;
    square.style.height = `${squareSize}px`;
    square.style.backgroundColor = fill;
    square.style.cursor = "pointer";

    square.addEventListener("mouseenter", (e) => {
        let statusSection = "";

        let totalVotes = calculateVotes(data.votes, 'total');
        let agreeVotes = calculateVotes(data.votes, 'agree');
        let disagreeVotes = calculateVotes(data.votes, 'disagree');
        let passVotes = calculateVotes(data.votes, 'pass');
        let isAgree = data.agreeRate > data.disagreeRate;

        if (data.isHighAlignment) {
            statusSection = `<div class="sm-tooltip-status ${isAgree ? 'agree' : 'disagree'}">${Math.round(
                isAgree ? data.agreeRate * 100 : data.disagreeRate * 100
            )}% voted ${isAgree ? 'agree' : 'disagree'}</div>`;
        } else if (data.isHighUncertainty) {
            statusSection = `<div class="sm-tooltip-status pass">${Math.round(
                data.passRate * 100
            )}% voted "unsure/pass"</div>`;
        } else {
            statusSection = `
                <div class="sm-tooltip-status">${Math.round(
                    data.agreeRate * 100
                )}% voted agree</div>
                <div class="sm-tooltip-status">${Math.round(
                    data.disagreeRate * 100
                )}% voted disagree</div>
            `;
        }

        let tooltipContent = `
            <div class="sm-tooltip-content">
                ${statusSection}
                <div class="sm-tooltip-comment">${data.text}</div>
                <div class="sm-tooltip-topics">${(() => {
                    const topicIndex = data;
                    return topicIndex >= 0
                        ? `${data.topics[topicIndex]} > ${data.subtopics[topicIndex]}`
                        : `${data.topics[0]} > ${data.subtopics[0]}`;
                })()}</div>
                <hr/>
                <div class="sm-tooltip-votes">
                    <div>${totalVotes.toLocaleString()} total votes</div>
                    <ul style="width: 200px;">
                        <li><span class="row">
                            <span>Agree</span>
                            <span class="count">${agreeVotes.toLocaleString()}</span>
                        </li>
                        <li><span class="row">
                            <span>Disagree</span>
                            <span class="count">${disagreeVotes.toLocaleString()}</span>
                        </li>
                        <li><span class="row">
                            <span>Unsure/Passed</span>
                            <span class="count">${passVotes.toLocaleString()}</span>
                        </li>
                    </ul>
                </div>
            </div>
        `;
        vizTooltip.show(
            tooltipContent,
            e.clientX,
            e.clientY,
            "for-statement"
        );
    });

    square.addEventListener("mousemove", (e) => {
        vizTooltip.move(e.clientX, e.clientY);
    });

    square.addEventListener("mouseleave", () => {
        vizTooltip.hide();
    });

    el.appendChild(square);
}
