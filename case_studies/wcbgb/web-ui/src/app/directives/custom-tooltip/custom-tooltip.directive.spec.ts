import { Component, TemplateRef, ViewChild } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Overlay } from '@angular/cdk/overlay';
import { By } from '@angular/platform-browser';
import { CustomTooltipDirective } from './custom-tooltip.directive';

// test host component
@Component({
  template: `
    <div [customTooltip]="tooltipTemplate">Host element</div>
    <ng-template #tooltipTemplate>Tooltip content</ng-template>
  `,
})
class TestComponent {
  @ViewChild('tooltipTemplate') tooltipTemplate!: TemplateRef<any>;
}

describe('CustomTooltipDirective', () => {
  let fixture: ComponentFixture<TestComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CustomTooltipDirective],
      declarations: [TestComponent],
      providers: [Overlay],
    }).compileComponents();

    fixture = TestBed.createComponent(TestComponent);
    fixture.detectChanges();
  });

  it('should create an instance of the directive with TemplateRef input', () => {
    const directiveEl = fixture.debugElement.query(By.directive(CustomTooltipDirective));
    expect(directiveEl).not.toBeNull(); // check that the directive is applied
    const directiveInstance = directiveEl.injector.get(CustomTooltipDirective);
    expect(directiveInstance).toBeTruthy(); // verify the instance exists
    expect(directiveInstance.content).toBeDefined(); // verify TemplateRef input is set
  });
});
