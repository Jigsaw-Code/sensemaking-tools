import { CommonModule } from '@angular/common';
import {
  Component,
  CUSTOM_ELEMENTS_SCHEMA,
  Input,
  ViewChild,
  ElementRef,
  AfterViewInit,
} from '@angular/core';
import '@conversationai/sensemaker-visualizations';

@Component({
  selector: 'app-sensemaking-chart-wrapper',
  standalone: true,
  imports: [CommonModule],
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
  template: `
    <div class="chart-container">
      <sensemaker-chart
        #sensemakingChartEl
        [attr.id]="chartId"
        [attr.chart-type]="chartType"
        [attr.view]="view"
        [attr.topic-filter]="topicFilter"
        [attr.colors]="colors?.length ? (colors | json) : null"
      ></sensemaker-chart>
    </div>
  `,
  styles: [
    `
      .chart-container {
        width: 100%;
        height: 100%;
      }
    `,
  ],
})
export class SensemakingChartWrapperComponent implements AfterViewInit {
  @ViewChild('sensemakingChartEl') chartElementRef!: ElementRef<
    HTMLElement & {
      data?: any;
      summaryData?: any;
    }
  >;

  @Input() chartId: string = '';
  @Input() chartType: string = 'topics-distribution';
  @Input() view: string = 'cluster';
  @Input() topicFilter: string = '';
  @Input() colors: string[] = [];
  @Input() data: any;
  @Input() summaryData: any;

  ngAfterViewInit() {
    // Set the data directly on the web component
    if (this.chartElementRef?.nativeElement) {
      const chartElement = this.chartElementRef.nativeElement;

      // Set the main data
      chartElement.data = this.data;

      // Set the summary data
      chartElement.summaryData = this.summaryData;
    }
  }
}
