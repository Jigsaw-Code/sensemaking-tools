import { Component } from '@angular/core';
import { MatIconRegistry } from "@angular/material/icon";
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    RouterOutlet,
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  constructor(private matIconRegistry: MatIconRegistry) {
    // set defaults for all material icons:
    //   use symbols (instead of icons)
    //   use "outlined" variation
    this.matIconRegistry.setDefaultFontSetClass("material-symbols-outlined");
  }
}
