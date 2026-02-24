import { ComponentFixture, TestBed } from '@angular/core/testing';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';
import { MarkdownModule, MarkdownService } from 'ngx-markdown';
import { ReportComponent } from './report.component';

describe('ReportComponent', () => {
  let component: ReportComponent;
  let fixture: ComponentFixture<ReportComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [
        MarkdownModule.forRoot(),
        NoopAnimationsModule,
        ReportComponent,
      ],
      providers: [MarkdownService]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ReportComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
