import {Modal} from '@google/glue/lib/modal';

const enum Selectors {
  modal = '.glue-modal',
  trigger = '[data-modal-trigger]',
}

module ModalModule {
  function init(): void {
    const triggerButtons = document.querySelectorAll<HTMLElement>(Selectors.trigger);
    const modalsMap = new Map<HTMLElement, Modal>();

    triggerButtons.forEach(trigger => {
      const targetId = trigger.getAttribute('data-modal-target');
      const modalEl = targetId
        ? document.getElementById(targetId)
        : document.querySelector<HTMLElement>(Selectors.modal);

      if (modalEl) {
        let modal = modalsMap.get(modalEl);
        if (!modal) {
          modal = new Modal(modalEl, trigger);
          modalsMap.set(modalEl, modal);
          
          modalEl.addEventListener('click', (e: MouseEvent) => {
            if (e.target === modalEl) {
              modal!.close();
            }
          });
        }
        
        trigger.addEventListener('click', () => {
          modal!.open();
        });
      }
    });
  }

  init();
}
