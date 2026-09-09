# Implementation Plan: FAQs Section, Partner Logos Sizing, Core Capabilities Layout, and Partner Copy Updates

## Overview
This plan details the updates to the Jigsaw Sensemaking AI landing page:
1. Revert "How it Works" feature cards: moved feature names back above the icons as primary bold headlines, with action phrases below the icons, matching design specifications (`media_1788971975128.png` and `media_1788972438279.png`).
2. Add descriptive copy under the technology partner logos: "Sensemaking AI is also available through our technology partners, as part of their polling or public deliberation offerings. As we onboard new partners, we will update this site."
3. Tuned technology partner logo spacing: proper breathing room between title, logos, copy, and CTA section.
4. Updated CTA copy in "Bring Sensemaking AI to Your Community" to include "Jigsaw" in front of "Partner Program" and changed the button text to "Apply here".
5. Implemented the responsive "FAQs" section between JPP copy and Technology Partners.

## Changes Breakdown

### 1. How it Works Cards (`src/index.html` & `src/styles/app/components/core-capabilities-card/_core-capabilities-card.scss`)
- Positioned bold feature titles (`.app-core-capabilities-card__feature-name`) above the icons:
  - `font-size: 1.25rem` (`1.375rem` at `lg`), `font-weight: 700`, `line-height: 1.4`, `min-height: calc(1.25rem * 1.4 * 2)`.
  - Margin bottom spacer before image (`spacer(3)`).
- Card titles in order above circles:
  - **Card 1**: "Adaptive Interviewing"
  - **Card 2**: "Automated Reporting"
  - **Card 3**: "Predictive Agreement"
  - **Card 4**: "Panoramic Insights"
- Kept icons at `width: 240px`.
- Retained action phrase headings (`<h3>`) with `.glue-headline--headline-5` and descriptive copy below the icons:
  - **Card 1**: "Dig Deeper"
  - **Card 2**: "Extract Key Themes"
  - **Card 3**: "Uncover Consensus"
  - **Card 4**: "Surface Context"

### 2. Technology Partner Logos, Copy & Spacing (`src/index.html` & `src/styles/app/components/use-sensemaking/_use-sensemaking.scss`)
- Maintained title and logo spacing.
- Added descriptive copy under the logos:
  - `<p class="glue-body--large app-body--large glue-spacer-4-top glue-text-center">Sensemaking AI is also available through our technology partners, as part of their polling or public deliberation offerings. As we onboard new partners, we will update this site.</p>`
- Kept enlarged logo sizes (`max-width: 190px`, `max-height: 68px`) and generous `spacer(6)` column gap.

### 3. JPP & CTA Copy (`src/index.html`)
- Updated CTA question to: "Are you a policymaker or public official interested in using Sensemaking AI through our Jigsaw Partner Program?"
- Updated CTA button text to: "Apply here".

## Verification & Status
- Webpack compile passes with 0 warnings and 0 errors.
- Visual screenshots captured and verified across desktop and mobile viewports.
