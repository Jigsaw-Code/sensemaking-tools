import {Carousel} from '@google/glue/lib/carousel';
import events from '../events/events';

const enum Selectors {
  carousel = '.glue-carousel',
  nextBtn = '[data-carousel-next]',
}

module CarouselModule {
  const carousels: {[key: string]: Carousel} = {};

  function init(): void {
    const carouselEls: NodeListOf<HTMLElement> = document.querySelectorAll(Selectors.carousel);

    for (const el of carouselEls) addCarousel(el);
  }

  function addCarousel(el: HTMLElement): void {
    if (!el.id) el.id = `carousel-${Object.entries(carousels).length}`;
    const id = el.id;
    const options = {navigation: true, peekOut: false, cyclical: true};
    const newCarousel = new Carousel(el, options);
    bindEvents(newCarousel, id);
    carousels[id] = newCarousel;
  }

  function bindEvents(carousel: Carousel, carouselId: string): void {
    const nextBtns = carousel.root.querySelectorAll(Selectors.nextBtn);
    for (const btn of nextBtns) {
      btn.addEventListener('click', carousel.next.bind(carousel));
    }

    carousel.observer.listen('currentSlide', () => {
      const currentSlide = carousel.observer.data.currentSlide;
      const event = new CustomEvent('carouselSlideChanged', {
        detail: {carouselId, currentSlide},
      });
      events.dispatchEvent(event);
    });
  }

  init();
}
