# Implementation Plan: Site Sections Re-ordering & Inline Technology Partners

## Overview
This plan details the reordering and layout refinement of the landing page sections based on stakeholder guidance:
1. **Jigsaw Partner Program**: Retains all heavy text (introductory overview and process breakdown) in a two-column layout (`app-bg-white`).
2. **Technology Partners Logos**: Moved directly underneath the JPP description copy in the left column (not in their own separate section), left-aligned, enlarged for legibility, preceded by an "Our Partners" sub-headline and generous top spacing/padding.
3. **Civic Leader Testimonials Carousel**: Dedicated, visually distinct section (`app-bg-testimonials` matching `$bg-grey` / `#F2F2ED`) positioned between JPP and the Apply CTA banner. Features title "Testimonials: What Civic Leaders Are Saying", compact vertical card proportions (`min-height: 310px`), pure white elevated quote cards with subtle warm-neutral borders, small aesthetic circular navigation arrows (`40px` diameter, `18px` chevrons), and cyclical 5-slide navigation.
4. **Banner to Apply**: Dedicated CTA section ("Bring Sensemaking AI to Your Community") with the "Apply here" button placed directly after the Testimonials (`app-bg-grey`).
5. **FAQs**: Dedicated FAQ section with heading and 4 responsive Q&A items placed after the Apply banner (`app-bg-white`).

## Changes Breakdown

### 1. Section Layout in `src/index.html`
* **Section 1 (Partner Program)**:
  * Container: `<section class="use-sensemaking glue-fullbleed g-spacer-padding-7-top g-spacer-padding-4-bottom glue-page app-bg-white">`
  * Left Column (span-6): Introductory paragraphs, followed by `<h3 class="... glue-spacer-2-bottom glue-spacer-4-top">Our Partners</h3>`, followed by enlarged partner logos (`use-sensemaking__logos use-sensemaking__logos--inline`).
  * Right Column (span-6): Process breakdown with 5 bullet points.
* **Section 2 (Civic Leader Testimonials Carousel)**:
  * Container: `<section id="testimonials" class="testimonials glue-fullbleed g-spacer-padding-5-top g-spacer-padding-5-bottom glue-page app-bg-testimonials">`
  * Heading: "Testimonials: What Civic Leaders Are Saying"
  * Carousel: `.app-carousel--testimonials` with cyclical 5-slide quote cards and dot navigation.
  * Slides:
    1. Mayor Patricia Lock Dawson (Riverside, CA)
    2. City Controller Chris Hollins (Houston, TX)
    3. Mayor Berry Vrbanovic (Kitchener, Ontario, Canada)
    4. Mayor Stephanie Orman (Bentonville, AR)
    5. Michał Zorena (Deputy Director of the Department of Social Development, Gdansk, Poland)
* **Section 3 (Banner to Apply)**:
  * Container: `<section id="apply-banner" class="use-sensemaking glue-fullbleed g-spacer-padding-7-top g-spacer-padding-7-bottom glue-page app-bg-grey">`
  * Headline: "Bring Sensemaking AI to Your Community"
  * CTA Button: "Apply here" (modal trigger).
* **Section 4 (FAQs)**:
  * Container: `<section id="faqs" class="use-sensemaking glue-fullbleed g-spacer-padding-7-top g-spacer-padding-7-bottom glue-page app-bg-white">`
  * Headline: "FAQs"
  * 4 FAQ grid columns (`span-6` on desktop, `span-12` on mobile).

### 2. Styling in `src/styles/`
* `_use-sensemaking.scss`:
  * `.use-sensemaking__logo--small`: `max-width: 140px`, `max-height: 46px`.
  * `.use-sensemaking__logos--inline`: `margin-top: 1rem; gap: 1.5rem 2.5rem;`.
* `_colors.scss`:
  * `.app-bg-testimonials`: `background-color: #F2F2ED;` (matches `$bg-grey`)
* `_carousel.scss`:
  * `.app-carousel--testimonials`:
    * `.app-carousel__slide--testimonial`: rounded (`16px`), white background (`#FFFFFF`), subtle warm-neutral border (`1px solid rgba(189, 189, 186, 0.45)`), soft shadow, compact padding (`2rem 1.5rem` mobile, `2.75rem 4rem` desktop), `min-height: 310px`.
    * `.glue-carousel__button`: compact `40px` circular elevated buttons, `#FFFFFF` surface, `18px` chevrons, subtle border, smooth hover scale.
    * `.app-carousel__testimonial-card`: centered layout with max-width `800px`.
    * `.app-carousel__testimonial-quote-mark`: stylized accent quote mark (`3.25rem`).
    * Typography: 1.0625rem - 1.1875rem quote text, 500-weight author name, secondary color location.

## Verification & Status
* Webpack build verification (`npm run build` with 0 errors).
* Visual verification via headless Chrome screenshot confirming alignment, typography, and layout.
* Verified live on `http://localhost:8080`.


