import { Directive, Input, HostListener, TemplateRef, ViewContainerRef } from "@angular/core";
import { GlobalPositionStrategy, Overlay, OverlayRef } from "@angular/cdk/overlay";
import { TemplatePortal } from "@angular/cdk/portal";

@Directive({
  selector: "[customTooltip]",
  standalone: true
})
export class CustomTooltipDirective {
  @Input("customTooltip") content!: TemplateRef<any>;
  private overlayRef!: OverlayRef;
  private portal: TemplatePortal<any> | null = null;

  constructor(
    private overlay: Overlay,
    private viewContainerRef: ViewContainerRef
  ) {}

  @HostListener("mousemove", ["$event"])
  updatePosition(event: MouseEvent) {
    if(!this.overlayRef) {
      return;
    }

    const strategy = this.overlayRef
      .getConfig()
      .positionStrategy as GlobalPositionStrategy;

    // measure rendered tooltip
    const tooltipEl = this.overlayRef.overlayElement;
    const { width, height } = tooltipEl.getBoundingClientRect();
    const viewportWidth = window.innerWidth;

    // set distance for tooltip offset from mouse point
    const offset = 10;

    // horizontal position
    let xPos = event.clientX + offset;
    if(event.clientX + offset + width > viewportWidth) {
      // overflows screen's right edge → flip to left of cursor
      xPos = event.clientX - width - offset;
    }

    // vertical position
    let yPos = event.clientY - height - offset;
    if(yPos < 0) {
      // overflows screen's top edge → flip to below cursor
      yPos = event.clientY + offset;
    }

    strategy.left(`${xPos}px`);
    strategy.top(`${yPos}px`);
    this.overlayRef.updatePosition();
  }

  @HostListener("mouseenter", ["$event"])
  show(event: MouseEvent) {
    if(!this.overlayRef) {
      const positionStrategy = this.overlay
        .position()
        .global()
        // start somewhere offscreen; reposition immediately
        .left("0px")
        .top("0px");

      this.overlayRef = this.overlay.create({
        positionStrategy,
        scrollStrategy: this.overlay.scrollStrategies.reposition()
      });
    }

    this.portal = new TemplatePortal(
      this.content,
      this.viewContainerRef
    );
    this.overlayRef.attach(this.portal);

    // initial positioning
    this.updatePosition(event);
  }

  @HostListener("mouseleave")
  hide() {
    if(this.overlayRef) {
      // remove the overlay
      this.overlayRef.detach();
    }
  }
}
