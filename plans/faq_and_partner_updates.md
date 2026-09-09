# Implementation Plan: Site Sections Re-ordering & Inline Technology Partners

## Overview
This plan details the reordering and layout refinement of the landing page sections based on stakeholder guidance:
1. **Jigsaw Partner Program**: Retains all heavy text (introductory overview and process breakdown) in a two-column layout (`app-bg-white`).
2. **Technology Partners (Inline & Smaller)**: Moved WITHIN the Partner Program section below the heavy text. Scaled down, left-aligned, not centered, and styled as an integrated sub-block rather than a loud standalone section.
3. **Banner to Apply**: Dedicated CTA section ("Bring Sensemaking AI to Your Community") with the "Apply here" button placed directly after the Partner Program (`app-bg-grey`).
4. **FAQs**: Dedicated FAQ section with heading and 4 responsive Q&A items placed after the Apply banner (`app-bg-white`).
5. **Testimonials**: Kept intact / ignored per request.

## Changes Breakdown

### 1. Section Reordering in `src/index.html`
* **Section 1 (Partner Program + Inline Partners)**:
  * Container: `<section class="use-sensemaking glue-fullbleed g-spacer-padding-7-top g-spacer-padding-7-bottom glue-page app-bg-white">`
  * Heading: `Jigsaw Partner Program`
  * Left Column (span-6): Two introductory paragraphs with GitHub link.
  * Right Column (span-6): Process breakdown with 5 bullet points.
  * Inline Partners Sub-block (span-12):
    * Copy: "Sensemaking AI is also available through our technology partners, as part of their polling or public deliberation offerings. As we onboard new partners, we will update this site."
    * Logos: Left-aligned flex layout with reduced dimensions (`max-width: 120px`, `max-height: 40px`).
* **Section 2 (Banner to Apply)**:
  * Container: `<section id="apply-banner" class="use-sensemaking glue-fullbleed g-spacer-padding-7-top g-spacer-padding-7-bottom glue-page app-bg-grey">`
  * Headline: "Bring Sensemaking AI to Your Community"
  * Paragraph: "Are you a policymaker or public official interested in using Sensemaking AI through our Jigsaw Partner Program?"
  * CTA Button: "Apply here" (modal trigger).
* **Section 3 (FAQs)**:
  * Container: `<section id="faqs" class="use-sensemaking glue-fullbleed g-spacer-padding-7-top g-spacer-padding-7-bottom glue-page app-bg-white">`
  * Headline: "FAQs" (`glue-headline--headline-4 glue-text-center`)
  * 4 FAQ grid columns (`span-6` on desktop, `span-12` on mobile).

### 2. Styling Updates in `src/styles/app/components/use-sensemaking/_use-sensemaking.scss`
* Add `.use-sensemaking__partners-inline`:
  * Border-top subtle separator or spacing (`glue-spacer-5-top`).
  * Left-aligned copy and logos.
* Add `.use-sensemaking__logos--inline`:
  * `justify-content: flex-start` (left-aligned).
  * Reduced gap (`column gap: 4`, `row gap: 3`).
* Add `.use-sensemaking__logo--small`:
  * `max-width: 120px`, `max-height: 40px`.
  * Subtle hover opacity transition.

## Verification & Status
* Webpack build verification (`npm run build-production` with 0 errors).
* Visual verification across desktop and mobile.
* Verify local server and staging deployment.

