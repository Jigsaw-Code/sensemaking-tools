import { ComponentFixture, TestBed } from '@angular/core/testing';

import { StatementCardComponent } from './statement-card.component';

describe('StatementCardComponent', () => {
  let component: StatementCardComponent;
  let fixture: ComponentFixture<StatementCardComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [StatementCardComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(StatementCardComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
