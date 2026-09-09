# Implementation Plan: FAQs Section, Partner Logos Sizing, and Core Capabilities Layout Updates

## Overview
This plan details the updates to the Jigsaw Sensemaking AI landing page:
1. Re-structured "How it Works" feature cards: moved feature names below the icons as primary bold headlines with action phrases directly underneath as styled subtitles.
2. Tuned technology partner logo spacing: added padding between the "Our Technology Partners" title and logos, while reducing excess padding underneath the logos before the CTA section.
3. Updated CTA copy in "Bring Sensemaking AI to Your Community" to include "Jigsaw" in front of "Partner Program" and changed the button text to "Apply here".
4. Implemented the responsive "FAQs" section between JPP copy and Technology Partners.

## Changes Breakdown

### 1. How it Works Cards (`src/index.html` & `src/styles/app/components/core-capabilities-card/_core-capabilities-card.scss`)
- Positioned bold feature titles (`.app-core-capabilities-card__feature-name`) above the icons (`<h3>`):
  - `font-size: 1.25rem` (`1.375rem` at `lg`), `font-weight: 700`, `line-height: 1.4`, `min-height: calc(1.25rem * 1.4 * 2)`.
  - Margin bottom spacer before image (`spacer(3)`).
- Card titles in order above circles:
  - **Card 1**: "Adaptive Interviewing"
  - **Card 2**: "Automated Reporting"
  - **Card 3**: "Predictive Agreement"
  - **Card 4**: "Panoramic Insights"
- Kept icons at `width: 240px`.
- Retained action phrase headings (`<h4>`) with `.glue-headline--headline-5` and descriptive copy below the icons:
  - **Card 1**: "Dig Deeper"
  - **Card 2**: "Extract Key Themes"
  - **Card 3**: "Uncover Consensus"
  - **Card 4**: "Surface Context"

### 2. Technology Partner Logos & Spacing (`src/index.html` & `src/styles/app/components/use-sensemaking/_use-sensemaking.scss`)
- Adjusted padding below the "Our Technology Partners" title (`glue-spacer-4-bottom` on `<h3>`, `spacer(4)` top margin on logos container).
- Added bottom padding/margin under partner logos: added `@include glue-spacer-mixins.spacer(4, margin, bottom);` to `.use-sensemaking__logos` to provide balanced breathing room above the CTA section.
- Maintained enlarged logo sizing (`max-width: 190px`, `max-height: 68px`) and generous `spacer(6)` column gap.

### 3. JPP & CTA Copy (`src/index.html`)
- Updated CTA question to: "Are you a policymaker or public official interested in using Sensemaking AI through our Jigsaw Partner Program?"
- Updated CTA button text to: "Apply here".

## Verification & Status
- Webpack compile passes with 0 warnings and 0 errors.
- Visual screenshots captured and verified across desktop and mobile viewports.
