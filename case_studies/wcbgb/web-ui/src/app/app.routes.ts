import { Routes } from '@angular/router';
import { ReportComponent } from './pages/report/report.component';

export const routes: Routes = [
  // Wildcard route is necessary for building "single HTML file" report.
  // Do not change path to anything else.
  { path: "**", component: ReportComponent },
];
